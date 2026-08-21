"""FedSCVI: A Flower for federated single cell variational inference."""

# Standard library
import gc
from pathlib import Path

# Third-party libraries
import anndata
import torch
import numpy as np

# Flower Message API imports
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

# Local helper functions used by the client
from app.task import (
    create_scvi_model,
    get_architecture,
    get_loss,
    get_weights,
    load_local_data_simulation,
    read_json,
    set_weights,
)

# Use higher precision matrix multiplications when possible
# (can slightly improve training stability/performance)
torch.set_float32_matmul_precision("high")

# Create the Flower client application
app = ClientApp()

# ---------------------------------------------------------------------
# Differential privacy helper (used only when strategy == "privacy")
# ---------------------------------------------------------------------

def add_dp_noise(weights, noise_multiplier, clip_norm):
    """Clip each layer's norm, then add calibrated Gaussian noise."""
    clipped = []
    for w in weights:
        norm = np.linalg.norm(w)
        factor = min(1.0, clip_norm / (norm + 1e-8))
        clipped.append(w * factor)

    noisy = [
        w + np.random.normal(0, noise_multiplier * clip_norm, w.shape)
        for w in clipped
    ]
    return noisy

def add_dp_noise_targeted(weights, noise_multiplier, clip_norm, target_layers):
    result = []
    for i, w in enumerate(weights):
        if i in target_layers:
            norm = np.linalg.norm(w)
            factor = min(1.0, clip_norm / (norm + 1e-8))
            w_clipped = w * factor
            w_noisy = w_clipped + np.random.normal(0, noise_multiplier * clip_norm, w.shape)
            result.append(w_noisy)
        else:
            result.append(w)  # invariato
    return result

# ---------------------------------------------------------------------
# Helper function: load local client state
# ---------------------------------------------------------------------

def _load_client_state(context: Context):
    """
    Load client-specific data and initialize the local SCVI model.

    Each Flower client corresponds to one data partition
    (for example, one sequencing technology).

    Steps:
    1. Read client ID from Flower context
    2. Load local train/validation datasets
    3. Restrict genes to the selected HVG list
    4. Build a local SCVI model
    """

    client_id = context.node_config["partition-id"]

    # Build client folder path using partition id
    client_data_folder = Path(
        context.run_config["client_folder_path"].format(
            partition_id=client_id
        )
    )

    # ---------------------------------------------------------
    # Load metadata files
    # ---------------------------------------------------------

    # Top highly-variable genes used in training
    hvg_file_path = client_data_folder / context.run_config["top2k_genes_file"]
    hvg_list = read_json(hvg_file_path)

    # Full list of possible technologies / batches
    batch_file_path = client_data_folder / context.run_config["all_techs_file"]
    batch_list = read_json(batch_file_path)

    # ---------------------------------------------------------
    # Load local train / validation data
    # ---------------------------------------------------------

    adata_local_train = anndata.read_h5ad(client_data_folder / "train.h5ad")
    adata_local_train = adata_local_train[:, hvg_list].copy()

    adata_local_valid = anndata.read_h5ad(client_data_folder / "valid.h5ad")
    adata_local_valid = adata_local_valid[:, hvg_list].copy()

    # # ---------------------------------------------------------
    # # THIS BLOCK IS ONLY FOR TESTING AND WILL BE REMOVED LATER
    # # ---------------------------------------------------------

    # adata_local_train = load_local_data_simulation(
    #     client_id,
    #     "data_centralized/pancreas_train.h5ad",
    # )

    # adata_local_valid = load_local_data_simulation(
    #     client_id,
    #     "data_centralized/pancreas_valid.h5ad",
    # )

    # adata_local_train = adata_local_train[:, hvg_list].copy()
    # adata_local_valid = adata_local_valid[:, hvg_list].copy()

    # ---------------------------------------------------------
    # Build local SCVI model
    # ---------------------------------------------------------

    arch_cfg = get_architecture(context.run_config)

    scvi_model = create_scvi_model(
        adata_local_train,
        batch_list,
        arch_cfg,
    )

    return client_id, adata_local_train, adata_local_valid, scvi_model

# ---------------------------------------------------------------------
# Training endpoint
# ---------------------------------------------------------------------

@app.train()
def train(msg: Message, context: Context) -> Message:
    """
    Run one round of local client training.

    Flower server sends:
      - current global model weights
      - training config (number of local epochs)

    Client returns:
      - updated local model weights
      - number of local training examples
    """

    # Load local data + local model
    client_id, adata_local_train, _, scvi_model = _load_client_state(context)

    # Number of local epochs chosen by the server
    num_local_epochs = int(msg.content["config"]["num_local_epochs"])

    # Load global weights received from the server
    set_weights(
        scvi_model,
        msg.content["arrays"].to_numpy_ndarrays(),
    )

    # ---------------------------------------------------------
    # Local training step
    # ---------------------------------------------------------

    scvi_model.train(
        max_epochs=num_local_epochs,
        train_size=1.0,
    )

    # Extract updated model weights after local training
    updated_weights = get_weights(scvi_model)

    # Apply differential privacy only for the "privacy" strategy
    strategy_name = context.run_config.get("strategy", "baseline")
    if strategy_name == "privacy":
        noise_multiplier = float(context.run_config.get("dp_noise_multiplier", 0.0))
        print("USING NOISE ", noise_multiplier)
        clip_norm = float(context.run_config.get("dp_clip_norm", 1.0))
        if noise_multiplier > 0.0:
            #updated_weights = add_dp_noise(updated_weights, noise_multiplier, clip_norm)
            updated_weights = add_dp_noise_targeted(updated_weights, noise_multiplier, clip_norm, target_layers=[6, 18, 24])

    # Prepare reply to server
    content = RecordDict(
        {
            "arrays": ArrayRecord(
                numpy_ndarrays=updated_weights
            ),
            "metrics": MetricRecord(
                {
                    # Used for weighted averaging on the server (FedAvg)
                    "num-examples": int(adata_local_train.n_obs),
                    "client_id": int(client_id),
                }
            ),
        }
    )

    # ---------------------------------------------------------
    # Free memory after local training
    # ---------------------------------------------------------

    del scvi_model
    del adata_local_train
    del updated_weights

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return Message(content=content, reply_to=msg)


# ---------------------------------------------------------------------
# Evaluation endpoint
# ---------------------------------------------------------------------

@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    """
    Evaluate current global model on local client data.

    The server sends global weights.
    The client computes:

      - train_loss
      - valid_loss

    and sends metrics back to the server.
    """

    # Load local data + model
    client_id, adata_local_train, adata_local_valid, scvi_model = _load_client_state(
        context
    )

    # Load current global model weights
    set_weights(
        scvi_model,
        msg.content["arrays"].to_numpy_ndarrays(),
    )

    print(
        f"Client ID: {client_id} | "
        f"Local valid data shape: {adata_local_valid.shape}"
    )

    # Compute local losses 

    train_loss = get_loss(scvi_model, adata_local_train)
    valid_loss = get_loss(scvi_model, adata_local_valid)
    

    # Reply to server with metrics only
    content = RecordDict(
        {
            "metrics": MetricRecord(
                {
                    # Used for weighted averaging
                    "num-examples": int(adata_local_valid.n_obs),

                    # Raw (depth-sensitive) losses -- kept for backward
                    # compatibility with existing strategies/plots
                    "train_loss": float(train_loss),
                    "valid_loss": float(valid_loss),

                    # Alias used by some strategies
                    "eval_loss": float(valid_loss),

                    # Useful for debugging
                    "client_id": int(client_id),
                }
            )
        }
    )

    # ---------------------------------------------------------
    # Free memory after evaluation
    # ---------------------------------------------------------

    del scvi_model
    del adata_local_train
    del adata_local_valid

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return Message(content=content, reply_to=msg)
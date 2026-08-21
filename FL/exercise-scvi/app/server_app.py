"""FedSCVI: A Flower for federated single cell variational inference."""

from pathlib import Path

from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg
from app.fairness.fairness_strategy import (
    FedAvgAdaptiveQFFL, FedAvgQFFL, FedAvgUniform)
from app.fairness.availability_strategy import FedAvgQFFLAvailability
from app.custom_strategy import (
    FedAvgSaveModel,
    FedAvgSaveModelPlotLosses,
    FedAvgSaveModelPlotLossesEarlyStopping,
    EarlyStopException,
    FedAvgSaveClientUpdates
)
from app.task import (
    create_dummy_adata_from_hvg,
    create_scvi_model,
    get_architecture,
    get_weights,
    make_on_final_arrays,
    read_json,
    set_weights,
)

app = ServerApp()


def _load_fairness_strategy(context: Context, num_rounds: int, model_file_path: str, loss_plot_path: str, scvi_model):
    fairness_strategy_name = context.run_config.get("fairness_strategy_name")
    fairness_q = float(context.run_config.get("fairness_q", 1.0))
    fairness_log_dir = context.run_config.get("fairness_log_dir", ".")
    # only used by strategy_name == "adaptive_qffl"
    fairness_q_alpha = float(context.run_config.get("fairness_q_alpha", 0.5))
    fairness_q_min = float(context.run_config.get("fairness_q_min", 0.0))
    fairness_q_max = float(context.run_config.get("fairness_q_max", 5.0))

    # Namespace outputs by strategy_name (and, for qffl, by q, weight_metric,
    # and seed) so repeated runs with different settings don't overwrite
    # each other's saved model / loss plot / fairness logs.
    run_tag = fairness_strategy_name
    if fairness_strategy_name == "qffl":
        run_tag = f"qffl_q{str(fairness_q).replace('.', 'p')}"

    model_file_path = f"{model_file_path}_{run_tag}"
    loss_plot_path = f"./fairness_results/{model_file_path}/{loss_plot_path}"
    loss_numpy_path = f"./fairness_results/{model_file_path}/global_loss.npy"
    fairness_log_dir = f"./fairness_results/{model_file_path}/"

    on_final = make_on_final_arrays(
            model=scvi_model,
            set_weights=set_weights,
            save_path=fairness_log_dir,
        )

    if fairness_strategy_name == "fedavg_ploss":
        # Baseline: standard num-examples-weighted FedAvg
        strategy = FedAvgSaveModelPlotLosses(
            num_rounds=num_rounds,
            on_final_arrays=on_final,
            fraction_train=1.0,
            fraction_evaluate=1.0,
            weighted_by_key="num-examples",
            loss_history_path=None,
            loss_numpy_path=loss_numpy_path,
            loss_plot_path=loss_plot_path,
        )

    elif fairness_strategy_name == "uniform":
        # Fairness baseline: equal weight per client in aggregate_train
        strategy = FedAvgUniform(
            num_rounds=num_rounds,
            on_final_arrays=on_final,
            fraction_train=1.0,
            fraction_evaluate=1.0,
            weighted_by_key="num-examples",
            loss_history_path=None,
            loss_numpy_path=loss_numpy_path,
            loss_plot_path=loss_plot_path,
            fairness_log_dir=fairness_log_dir,
        )

    elif fairness_strategy_name == "qffl":
        # q-Fair Federated Learning: reweight by (cached) client loss ** q
        strategy = FedAvgQFFL(
            q=fairness_q,
            num_rounds=num_rounds,
            on_final_arrays=on_final,
            fraction_train=1.0,
            fraction_evaluate=1.0,
            weighted_by_key="num-examples",
            loss_history_path=None,
            loss_numpy_path=loss_numpy_path,
            loss_plot_path=loss_plot_path,
            fairness_log_dir=fairness_log_dir,
        )

    elif fairness_strategy_name == "adaptive_qffl":
        # F^3-inspired: q adapts every round based on the rate of change
        # of Jain's fairness index (see fairness_strategy.py docstring)
        strategy = FedAvgAdaptiveQFFL(
            q=fairness_q,
            alpha=fairness_q_alpha,
            q_min=fairness_q_min,
            q_max=fairness_q_max,
            num_rounds=num_rounds,
            on_final_arrays=on_final,
            fraction_train=1.0,
            fraction_evaluate=1.0,
            weighted_by_key="num-examples",
            loss_history_path=None,
            loss_plot_path=loss_plot_path,
            loss_numpy_path=loss_numpy_path,
            fairness_log_dir=fairness_log_dir,
        )

    else:
        raise ValueError(
            f"Unknown strategy_name: {fairness_strategy_name!r}. "
            "Expected one of: 'fedavg_ploss', 'uniform', 'qffl', 'adaptive_qffl', 'early_stopping'."
        )

    return strategy


def _load_availability_strategy(context: Context, num_rounds: int, model_file_path: str, loss_plot_path: str, scvi_model):
    participation_regime = context.run_config.get("participation_regime", "full")
    availability_q = float(context.run_config.get("availability_q", 0.0))
    availability_p = float(context.run_config.get("availability_p", 0.7))
    availability_size_threshold = int(context.run_config.get("availability_size_threshold", 800))
    availability_p_small = float(context.run_config.get("availability_p_small", 0.5))
    availability_p_large = float(context.run_config.get("availability_p_large", 0.9))
    availability_seed = int(context.run_config.get("availability_seed", 42))
    availability_bootstrap_rounds = int(context.run_config.get("availability_bootstrap_rounds", 1))

    # Namespace outputs by regime + q so the grid of runs doesn't collide.
    q_tag = f"q{str(availability_q).replace('.', 'p')}"
    run_tag = f"avail_{participation_regime}_{q_tag}"

    model_file_path = f"{model_file_path}_{run_tag}"
    loss_plot_path = f"./pp_results/{model_file_path}/{loss_plot_path}"
    loss_numpy_path = f"./pp_results/{model_file_path}/global_loss.npy"
    fairness_log_dir = f"./pp_results/{model_file_path}/"
    participation_log_path = f"{fairness_log_dir}/participation_log.csv"

    on_final = make_on_final_arrays(
        model=scvi_model,
        set_weights=set_weights,
        save_path=fairness_log_dir,
    )

    strategy = FedAvgQFFLAvailability(
        q=availability_q,
        participation_regime=participation_regime,
        p_available=availability_p,
        size_threshold=availability_size_threshold,
        p_small=availability_p_small,
        p_large=availability_p_large,
        availability_seed=availability_seed,
        bootstrap_rounds=availability_bootstrap_rounds,
        participation_log_path=participation_log_path,
        num_rounds=num_rounds,
        on_final_arrays=on_final,
        fraction_train=1.0,
        fraction_evaluate=1.0,
        weighted_by_key="num-examples",
        loss_history_path=None,
        loss_numpy_path=loss_numpy_path,
        loss_plot_path=loss_plot_path,
        fairness_log_dir=fairness_log_dir,
        fairness_log_tag=run_tag,
    )

    return strategy


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # ============================================================
    # 1. Read configuration parameters
    # ============================================================

    # Training hyperparameters
    num_rounds = int(context.run_config["num_rounds"])
    num_local_epochs = int(context.run_config["num_local_epochs"])
    early_stopping_patience = int(context.run_config.get("early_stopping_patience")) # Only used by Strategy 3
    early_stopping_min_delta = float(context.run_config.get("early_stopping_min_delta")) # Only used by Strategy 3

    # File paths
    server_folder = Path(context.run_config["server_folder_path"])
    top2k_genes_file_path = server_folder / context.run_config["top2k_genes_file"]
    all_techs_file_path = server_folder / context.run_config["all_techs_file"]
    model_file_path = context.run_config["model_file_path"]
    loss_plot_path = context.run_config.get("loss_plot_path", "loss_plot.png") # Only used by Strategies 2 and 3

    # ============================================================
    # 2. Build the initial global SCVI model
    # ============================================================
    
    # Load HVG list and techs list
    hvg_list = read_json(top2k_genes_file_path)
    batch_list = read_json(all_techs_file_path)
    
    # Note that initialization is performed through a dummy AnnData with the correct genes and batches
    adata_dummy = create_dummy_adata_from_hvg(hvg_list, batch_list)
    arch_cfg = get_architecture(context.run_config)
    scvi_model = create_scvi_model(adata_dummy, batch_list, arch_cfg)

    # Get initial global model weights as NumPy arrays to send to clients
    initial_arrays = ArrayRecord(numpy_ndarrays=get_weights(scvi_model))


    # ============================================================
    # 3. Strategy selection: uncomment exactly one strategy to run
    # ============================================================

    # Utility function for saving model 
    on_final = make_on_final_arrays(
        model=scvi_model,
        set_weights=set_weights,
        save_path=model_file_path,
    )

    # Strategy config
    strategy_name = context.run_config.get("strategy", "baseline")
    if strategy_name == "privacy":
        strategy = FedAvgSaveClientUpdates(
            num_rounds=num_rounds,         
            on_final_arrays=on_final,
            fraction_train=1.0,
            fraction_evaluate=1.0,
            weighted_by_key="num-examples",
            loss_history_path=None,
            loss_plot_path=loss_plot_path,
            early_stopping_patience=15, #5
            early_stopping_min_delta=1.0,
        )
    elif strategy_name == "fairness":
        strategy = _load_fairness_strategy(context, num_rounds, model_file_path, loss_plot_path, scvi_model)
    elif strategy_name == "availability":
        strategy = _load_availability_strategy(context, num_rounds, model_file_path, loss_plot_path, scvi_model)
    else:
        # Strategy 0: plain FedAvg
        #   - trains federated model
        #   - does not save final model
        #   - does not evaluate clients
        # strategy = FedAvg(
        #     fraction_train=1.0,
        #     fraction_evaluate=0.0,
        #     weighted_by_key="num-examples",
        # )

        # # Strategy 1: FedAvg + save final model
        # #   - saves final aggregated SCVI model
        # strategy = FedAvgSaveModel(
        #     num_rounds=num_rounds,
        #     on_final_arrays=on_final,
        #     fraction_train=1.0,
        #     fraction_evaluate=0.0,
        #     weighted_by_key="num-examples",
        # )

        # # Strategy 2: FedAvg + train/valid loss plot 
        # #   - tracks train and validation losses
        strategy = FedAvgSaveModelPlotLosses(
            num_rounds=num_rounds,
            on_final_arrays=on_final,
            fraction_train=1.0,
            fraction_evaluate=1.0,
            weighted_by_key="num-examples",
            loss_history_path=None,
            loss_plot_path=loss_plot_path,
        )

        # # Strategy 3: FedAvg + train/valid loss plot + early stopping
        # #   - tracks train and validation losses
        # #   - optionally stops early
        # strategy = FedAvgSaveModelPlotLossesEarlyStopping(
        #     num_rounds=num_rounds,
        #     on_final_arrays=on_final,
        #     fraction_train=1.0,
        #     fraction_evaluate=1.0,
        #     weighted_by_key="num-examples",
        #     loss_history_path=None,
        #     loss_plot_path=loss_plot_path,
        #     early_stopping_patience=early_stopping_patience,
        #     early_stopping_min_delta=early_stopping_min_delta,
        # )


    # ============================================================
    # 4. Start federated training
    # ============================================================

    try:
        strategy.start(
            grid=grid,
            initial_arrays=initial_arrays,
            train_config=ConfigRecord({"num_local_epochs": num_local_epochs}),
            num_rounds=num_rounds,
            evaluate_fn=None,
        )
    # This is raised by Strategy 3 when early stopping criteria are met
    except EarlyStopException:
        print("[EARLY STOP] Flower run stopped early.")
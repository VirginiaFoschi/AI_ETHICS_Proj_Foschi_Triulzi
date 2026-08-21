import csv
from pathlib import Path

import anndata
import numpy as np

NUM_CLIENTS = 7
CLIENT_FOLDER_TEMPLATE = "../data_federated/data_client_{partition_id}"
VALID_FILE = "valid.h5ad"

rows = []
for cid in range(NUM_CLIENTS):
    folder = Path(CLIENT_FOLDER_TEMPLATE.format(partition_id=cid))
    adata = anndata.read_h5ad(folder / VALID_FILE)

    lib_size = np.asarray(adata.X.sum(axis=1)).flatten()

    row = {
        "client_id": cid,
        "n_cells": int(adata.n_obs),
        "mean_lib_size": float(lib_size.mean()),
        "median_lib_size": float(np.median(lib_size)),
    }
    rows.append(row)
    print(
        f"client {cid}: n_cells={row['n_cells']}, "
        f"mean_lib_size={row['mean_lib_size']:.1f}, "
        f"median_lib_size={row['median_lib_size']:.1f}"
    )

with open("client_library_sizes.csv", "w", newline="") as f:
    writer = csv.DictWriter(
        f, fieldnames=["client_id", "n_cells", "mean_lib_size", "median_lib_size"]
    )
    writer.writeheader()
    writer.writerows(rows)

print("\nSaved client_library_sizes.csv")

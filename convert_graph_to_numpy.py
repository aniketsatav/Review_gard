"""
One-time conversion script.
Run locally (where torch_geometric is installed) to export:
  graph_data.pt  ->  graph_x.npy + graph_edges.npz
  gnn_model_weighted.pt -> gnn_weights.npz
"""
import torch
import numpy as np
import scipy.sparse as sp
from pathlib import Path

BASE = Path(__file__).parent

# ── 1. Load graph ──────────────────────────────────────────────────────────────
print("Loading graph_data.pt ...")
data = torch.load(BASE / "graph_data.pt", map_location="cpu", weights_only=False)

x = data.x.numpy().astype(np.float32)       # (N, 7)
edge_index = data.edge_index.numpy()         # (2, E)
print(f"  Nodes: {x.shape[0]}, Edges: {edge_index.shape[1]}, Features: {x.shape[1]}")

# ── 2. Save node features ──────────────────────────────────────────────────────
np.save(BASE / "graph_x.npy", x)
print("  Saved graph_x.npy")

# ── 3. Build symmetric normalised adjacency with self-loops ───────────────────
#  A_hat = D^(-1/2) (A + I) D^(-1/2)
N = x.shape[0]
row, col = edge_index[0], edge_index[1]

# Build A + I (add self-loops)
row_sl = np.concatenate([row, np.arange(N)])
col_sl = np.concatenate([col, np.arange(N)])
vals   = np.ones(len(row_sl), dtype=np.float32)
A = sp.coo_matrix((vals, (row_sl, col_sl)), shape=(N, N)).tocsr()

# Degree matrix D^(-1/2)
deg = np.asarray(A.sum(axis=1)).flatten()
d_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0).astype(np.float32)

# Symmetric normalisation: A_hat = D^{-1/2} A D^{-1/2}
D_inv_sqrt = sp.diags(d_inv_sqrt)
A_hat = D_inv_sqrt @ A @ D_inv_sqrt

sp.save_npz(BASE / "graph_edges.npz", A_hat.astype(np.float32))
print("  Saved graph_edges.npz (normalised adjacency)")

# ── 4. Load model weights ──────────────────────────────────────────────────────
print("\nLoading gnn_model_weighted.pt ...")
state = torch.load(BASE / "gnn_model_weighted.pt", map_location="cpu", weights_only=True)
print("  Weight keys:", list(state.keys()))

weights = {k: v.numpy().astype(np.float32) for k, v in state.items()}
np.savez(BASE / "gnn_weights.npz", **weights)
print("  Saved gnn_weights.npz")

print("\nConversion complete. Files created:")
for f in ["graph_x.npy", "graph_edges.npz", "gnn_weights.npz"]:
    size_mb = (BASE / f).stat().st_size / 1024 / 1024
    print(f"  {f}: {size_mb:.1f} MB")

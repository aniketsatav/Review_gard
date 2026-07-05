"""
GNN inference — pure NumPy/SciPy implementation of the trained GCN.
No torch_geometric or CUDA required.  Works on any server including Render free tier.

Architecture (mirrors training):
  GCNConv(7 -> 64) -> ReLU -> Dropout(0.3)
  GCNConv(64 -> 32) -> ReLU
  Linear(32 -> 2) -> log_softmax

Graph stored as:
  graph_x.npy            — (N, 7) float32 node features
  graph_edges.npz        — (N, N) normalised adjacency  A_hat = D^-1/2 (A+I) D^-1/2
  gnn_weights.npz        — model weight arrays
  scaler.pkl             — StandardScaler fitted on training features
"""

import numpy as np
import scipy.sparse as sp
import pickle
import threading
from pathlib import Path

# ── File paths ────────────────────────────────────────────────────────────────
_BASE         = Path(__file__).parent
_X_PATH       = _BASE / "graph_x.npy"
_EDGES_PATH   = _BASE / "graph_edges.npz"
_WEIGHTS_PATH = _BASE / "gnn_weights.npz"
_SCALER_PATH  = _BASE / "scaler.pkl"

# ── Thread-safe singleton state ───────────────────────────────────────────────
_lock        = threading.Lock()
_x           = None      # (N, 7) float32
_A_hat       = None      # scipy sparse normalised adjacency
_w           = None      # dict of weight arrays
_scaler      = None
_initialized = False
_init_error  = None


# ── GCN math helpers ─────────────────────────────────────────────────────────

def _relu(x):
    return np.maximum(0.0, x)


def _log_softmax(x):
    x = x - x.max(axis=1, keepdims=True)
    log_exp_sum = np.log(np.exp(x).sum(axis=1, keepdims=True))
    return x - log_exp_sum


def _gcn_conv(A_hat, x, weight, bias):
    """
    GCNConv forward (no self-loops added here — already in A_hat):
      out = A_hat @ x @ W^T + b
    weight shape: (out_features, in_features)
    """
    # A_hat @ x  — sparse x dense
    ax = A_hat.dot(x)                        # (N, in_features)
    return ax @ weight.T + bias              # (N, out_features)


def _gcn_forward(A_hat, x, w):
    """Full 2-layer GCN forward pass — no dropout at inference."""
    h = _relu(_gcn_conv(A_hat, x, w["conv1.lin.weight"], w["conv1.bias"]))
    h = _relu(_gcn_conv(A_hat, h, w["conv2.lin.weight"], w["conv2.bias"]))
    logits = h @ w["classifier.weight"].T + w["classifier.bias"]
    return _log_softmax(logits)


# ── One-time loader ───────────────────────────────────────────────────────────

def _load_all():
    global _x, _A_hat, _w, _scaler, _initialized, _init_error
    with _lock:
        if _initialized:
            return
        try:
            if not _X_PATH.exists():
                raise FileNotFoundError(f"graph_x.npy not found at {_X_PATH}")
            if not _EDGES_PATH.exists():
                raise FileNotFoundError(f"graph_edges.npz not found at {_EDGES_PATH}")
            if not _WEIGHTS_PATH.exists():
                raise FileNotFoundError(f"gnn_weights.npz not found at {_WEIGHTS_PATH}")
            if not _SCALER_PATH.exists():
                raise FileNotFoundError(f"scaler.pkl not found at {_SCALER_PATH}")

            print("[GNN] Loading graph features ...", flush=True)
            _x = np.load(_X_PATH)                               # (N, 7) float32

            print("[GNN] Loading adjacency matrix ...", flush=True)
            _A_hat = sp.load_npz(_EDGES_PATH)                   # (N, N) sparse

            print("[GNN] Loading model weights ...", flush=True)
            npz = np.load(_WEIGHTS_PATH)
            _w = {k: npz[k].astype(np.float32) for k in npz.files}

            print("[GNN] Loading scaler ...", flush=True)
            with open(_SCALER_PATH, "rb") as f:
                _scaler = pickle.load(f)

            print(f"[GNN] Loaded. Nodes: {_x.shape[0]}, "
                  f"Edges: {_A_hat.nnz}, Weights: {list(_w.keys())}", flush=True)

        except Exception as exc:
            _init_error = str(exc)
            print(f"[GNN] Load failed: {_init_error}", flush=True)
        finally:
            _initialized = True


# ── Feature extractor (identical to training) ─────────────────────────────────

def _extract_features(text: str, rating: float, bert_score: float):
    words      = text.split()
    word_count = max(len(words), 1)
    text_len   = min(len(text) / 1000.0, 1.0)
    rating_norm = float(rating) / 5.0
    avg_wl     = np.mean([len(w) for w in words]) / 15.0 if words else 0.0
    excl       = min(text.count("!") / 10.0, 1.0)
    caps_ratio = sum(1 for w in words if w.isupper() and len(w) > 2) / word_count
    return [rating_norm, text_len, float(word_count),
            avg_wl, excl, caps_ratio, float(bert_score)]


# ── Normalised-adjacency augmentation for a new node ─────────────────────────

def _augment_adjacency(A_hat_base, x_base, new_feat_scaled):
    """
    Append one new node to the graph, connected to its 5 nearest
    existing nodes by bert_score (feature index 6).
    Returns augmented (A_hat_aug, x_aug).
    """
    N = x_base.shape[0]

    # --- find 5 nearest by bert_score ---
    bert_col      = x_base[:, 6]
    new_bert      = new_feat_scaled[0, 6]
    diffs         = np.abs(bert_col - new_bert)
    neighbors     = np.argsort(diffs)[:5]

    # --- build new edges: bidirectional + self-loop ---
    new_idx = N
    src = list(neighbors) + [new_idx] * len(neighbors) + [new_idx]
    dst = [new_idx] * len(neighbors) + list(neighbors) + [new_idx]
    new_rows = np.array(src, dtype=np.int32)
    new_cols = np.array(dst, dtype=np.int32)
    new_vals = np.ones(len(src), dtype=np.float32)

    # --- augmented adjacency (A + I, unnormalised) ---
    # Re-normalise by computing D^{-1/2}(A+I)D^{-1/2} on the augmented graph.
    # Existing A_hat already has self-loops folded in, so we only add the new
    # node's edges and re-normalise only the affected rows/cols for efficiency.
    # Full re-normalisation is exact but expensive for 40K nodes.
    # Instead, we compute the new D^{-1/2} for the new node and its neighbors.

    # Degrees in the original un-normalised adjacency (approximated from A_hat diagonal)
    # We add the new edges and compute degree for new node.
    new_degree = float(len(neighbors) + 1)   # self-loop + neighbors
    new_d_inv_sqrt = 1.0 / np.sqrt(new_degree)

    # Neighbour degrees increase by 1 each (they gain an edge to the new node).
    # We pull existing diagonal of A_hat to estimate old sqrt-degree:
    old_diag = np.array(A_hat_base.diagonal()).flatten()

    # Build augmented sparse A_hat as block-style extension
    # Row N (new node) and col N (new node) added to existing (N, N) A_hat
    aug_size = N + 1

    # Rows/cols for new node contribution
    # off-diagonal: new_d_inv_sqrt * old_d_inv_sqrt[neighbor] * weight
    # For simplicity we approximate old D^{-1/2} from diagonal (A_hat_ii = d_i^{-1}).
    old_d_inv_sqrt_neighbors = np.sqrt(old_diag[neighbors].clip(min=1e-6))

    new_row_vals = new_d_inv_sqrt * old_d_inv_sqrt_neighbors   # (5,)
    self_val     = new_d_inv_sqrt * new_d_inv_sqrt              # scalar

    # Construct the augmented A_hat as a scipy sparse matrix
    # Block: [ A_hat (N,N) | col_ext (N,1) ]
    #        [ row_ext(1,N)| self    (1,1)  ]

    # Column extension: new edges seen from existing nodes' perspective
    col_ext_rows = neighbors
    col_ext_vals = new_row_vals       # same edge weight by symmetry
    col_ext = sp.coo_matrix(
        (col_ext_vals, (col_ext_rows, np.zeros(len(neighbors), dtype=np.int32))),
        shape=(N, 1)
    ).tocsr()

    # Row extension: new node's outgoing edges
    row_ext = sp.coo_matrix(
        (new_row_vals, (np.zeros(len(neighbors), dtype=np.int32), neighbors)),
        shape=(1, N)
    ).tocsr()

    # Self-loop for new node
    self_mat = sp.csr_matrix(np.array([[self_val]], dtype=np.float32))

    # Assemble
    top    = sp.hstack([A_hat_base, col_ext])
    bottom = sp.hstack([row_ext, self_mat])
    A_aug  = sp.vstack([top, bottom]).tocsr()

    # Augmented feature matrix
    x_aug = np.vstack([x_base, new_feat_scaled])    # (N+1, 7)

    return A_aug, x_aug, new_idx


# ── Public API ────────────────────────────────────────────────────────────────

def get_gnn_score(text: str, rating: float = 3.0, bert_score: float = 0.5) -> float:
    """
    Runs the real GCN inference on the augmented graph.
    Returns fake probability in [0.0, 1.0].
    Falls back to bert_score if files are unavailable.
    """
    _load_all()

    if _init_error or _x is None or _A_hat is None or _w is None:
        return round(float(bert_score), 4)

    try:
        raw        = _extract_features(text, rating, bert_score)
        scaled     = _scaler.transform([raw]).astype(np.float32)   # (1, 7)

        A_aug, x_aug, new_idx = _augment_adjacency(_A_hat, _x, scaled)

        log_probs = _gcn_forward(A_aug, x_aug, _w)     # (N+1, 2)
        probs     = np.exp(log_probs[new_idx])           # (2,)
        fake_prob = float(probs[1])                      # index 1 = FAKE

        return round(fake_prob, 4)

    except Exception as e:
        print(f"[GNN] Inference error: {e}", flush=True)
        return round(float(bert_score), 4)


def gnn_status() -> dict:
    """Returns load status — expose via /api/gnn-status for debugging."""
    return {
        "initialized":  _initialized,
        "error":        _init_error,
        "model_loaded": _w is not None,
        "graph_loaded": _x is not None,
        "nodes": int(_x.shape[0]) if _x is not None else 0,
        "edges": int(_A_hat.nnz)  if _A_hat is not None else 0,
    }

# Mistake Log

### GNN polarity inversion causes universal UNVERIFIABLE verdict
**Problem:** Every review was classified as UNVERIFIABLE (divergence ~0.98). The `1.0 - gnn_score` inversion in `predict_review()` was correct for real GCN inference, but the GNN fallback (when torch_geometric is absent) returns `bert_score` directly. Inverting that made GNN = `1 - bert_score`, creating maximum disagreement between BERT and GNN.
**Context:** `app.py` `predict_review()`, deployed on Render where torch_geometric cannot be installed.
**Solution:** Replaced the inversion with direct pass-through (`gnn_corrected = gnn_score`) and fixed the GCN output index from `[0]` (GENUINE) to `[1]` (FAKE) so the real model also uses the same polarity convention.
**Date:** 2026-07-05

### torch_geometric breaks Render deployment startup
**Problem:** Importing `torch_geometric` at module level caused Render's Gunicorn process to crash on startup — `ModuleNotFoundError` before the app even started serving.
**Context:** `gnn_inference.py`, Render free-tier deployment.
**Solution:** Rewrote `gnn_inference.py` entirely using `numpy` + `scipy.sparse`. Exported `graph_data.pt` → `graph_x.npy` + `graph_edges.npz` + `gnn_weights.npz` (1.5 MB total) via a local conversion script. Real GCN inference now works on Render with ~0.17s per call after warm-up.
**Date:** 2026-07-05

### HuggingFace Serverless Inference API returns 404 despite model being uploaded
**Problem:** `api-inference.huggingface.co` returned 404 for the model even though `model.safetensors` was correctly uploaded. The model page showed "This model isn't deployed by any Inference Provider."
**Context:** `app.py` `infer_with_hf_api()`, Render production, `MODEL_BACKEND=hf_api`.
**Solution:** Added a fallback that, on 404, calls the Gradio Space (`ParagR24/reviewguard-demo`) via `gradio_client`. Also updated primary endpoint to `router.huggingface.co`.
**Date:** 2026-07-05

### scikit-learn missing from requirements causes unpickling failure
**Problem:** `scaler.pkl` was saved using `scikit-learn`. When the new `gnn_inference.py` tried to load it using `pickle.load(f)`, it threw `ModuleNotFoundError: No module named 'sklearn'` on Render.
**Context:** Render deployment, loading models.
**Solution:** Added `scikit-learn>=1.3.0,<2.0.0` to `requirements.txt`.
**Date:** 2026-07-05

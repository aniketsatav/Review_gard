# ReviewGuard AI - Comprehensive Application Analysis

## 1. Overview and Architecture Summary

ReviewGuard AI is a full-stack machine learning web application designed to detect fake, deceptive, and synthetic product reviews on e-commerce platforms (such as Amazon and Flipkart). The application uses a tri-hybrid detection engine combining:
1. Deep Contextual NLP via Fine-Tuned BERT (Weight: 65%)
2. Graph Neural Networks (GCN) via PyTorch Geometric (Weight: 35%)
3. Rule-Based Heuristic Signal Analysis (Dynamic boost up to +0.50)

The application is built using Flask for the HTTP server, PyTorch and PyTorch Geometric for machine learning inference, and Bootstrap/Vanilla CSS with HTML templates for the user interface.

---

## 2. Machine Learning Pipeline and Ensemble Model

### 2.1 Fine-Tuned BERT Model
- Primary textual feature analyzer.
- Supports three deployment backends:
  - `local`: Direct inference using local PyTorch weights stored in `./bert_model`.
  - `pipeline`: Remote Hugging Face model loaded directly via the Hugging Face `transformers` library.
  - `hf_api`: Hugging Face Serverless Inference API with automatic cold-start retries and backoff logic.
- Output: Produces `bert_fake_score` ranging from 0.0 (Genuine) to 1.0 (Fake).

### 2.2 Graph Neural Network (GCN) Engine
- Implemented in `gnn_inference.py`.
- Architecture: 2-layer Graph Convolutional Network (`GCNConv(7, 64) -> Dropout(0.3) -> GCNConv(64, 32) -> Linear(32, 2)`).
- Graph Base: Pre-trained graph containing 40,432 review nodes and 521,140 edges (`graph_data.pt`).
- Node Feature Vector (7 dimensions):
  1. `rating_norm`: Normalized star rating (1.0 to 5.0 scaled to 0.2 - 1.0).
  2. `text_length`: Total character count.
  3. `word_count`: Total word count.
  4. `avg_word_length`: Average character length per word.
  5. `exclamation`: Count of exclamation marks (`!`).
  6. `caps_ratio`: Uppercase character ratio.
  7. `bert_score`: BERT model confidence score.
- Dynamic Graph Augmentation: During inference, new reviews are added as a new node, scaled via `scaler.pkl`, and connected to their 5 nearest neighbor nodes based on `bert_score` similarity.

### 2.3 Heuristic Engine
- Evaluates non-verbal text patterns:
  - Exclamation mark spam (>= 3 exclamation marks: +0.25).
  - Vague superlative word overload (words like amazing, best, perfect, love: +0.20).
  - High word repetition (same non-stopwords used >= 4 times: +0.30).
  - Duplicate copy-paste sentences (+0.15).
  - ALL CAPS overuse (>30% uppercase words: +0.15).
  - Complete lack of criticism in long reviews (+0.10).
- Max cumulative heuristic boost is capped at +0.50.

### 2.4 Verdict Determination and Divergence Handling
- Fused Score Calculation: `fused_score = 0.65 * bert_fake_score + 0.35 * (1.0 - gnn_score)`
- Final Score: `final_score = min(fused_score + heuristic_boost, 1.0)`
- Verdict Thresholds:
  - Score > 0.35: `FAKE`
  - Score <= 0.35: `GENUINE`
- Divergence Check: If `|bert_fake_score - gnn_corrected| >= 0.35`, the model flags the prediction as `UNVERIFIABLE` due to high model conflict, requiring manual review.

---

## 3. Web Scraping Module

The application includes real-time product review scrapers for e-commerce sites (`app.py`):
- Amazon Scraper (`scrape_amazon_reviews`):
  - Converts product URLs into standardized review page URLs (`/product-reviews/ASIN/`).
  - Extracts review body elements (`span[data-hook='review-body']`).
  - Gracefully handles 503 rate-limiting and CAPTCHA block pages.
- Flipkart Scraper (`scrape_flipkart_reviews`):
  - Parses Flipkart product DOM structures using fallback CSS selectors (`div.ZmyHeo`, `div.t-ZTKy`, `p.z9E0IG`).
- Bulk Scraper (`/scrape` and `/bulk-analyze`):
  - Evaluates up to 15 reviews per batch.
  - Computes product-level Trust Scores and overall risk classifications (`LOW RISK`, `MODERATE RISK`, `HIGH RISK`).

---

## 4. Web Application Routes and API Endpoints

### HTML UI Routes
- `/`: Landing page (`landing.html`).
- `/workbench`: Primary review analysis UI (`index.html`).
- `/reports`: Evaluation and analytics dashboard (`reports.html`).
- `/docs`: Project documentation and API reference (`docs.html`).
- `/settings`: Configuration and environment panel (`settings.html`).

### JSON REST API Endpoints
- `POST /predict`: Analyzes a single review text and optional star rating.
- `POST /scrape`: Fetches and analyzes reviews directly from an Amazon or Flipkart product link.
- `POST /bulk-analyze`: Analyzes multiple user-pasted reviews separated by blank lines.
- `GET /health`: Healthcheck endpoint reporting active backend and model configurations.
- `GET /api/system-status`: System metrics, dataset size, model accuracy, and precision stats.
- `GET /api/gnn-status`: PyTorch Geometric initialization state, graph node/edge counts, and memory status.

---

## 5. Deployment and Environment Configuration

Environment variables used by `app.py`:
- `MODEL_BACKEND`: Forced selection of model backend (`local`, `pipeline`, `hf_api`, `auto`).
- `HF_MODEL_ID`: Hugging Face model repository path.
- `HF_TOKEN`: Hugging Face API authentication token.
- `RENDER` / `RENDER_SERVICE_ID`: Render cloud detection flag. Defaults backend to `hf_api` on Render.
- `PORT`: Server port (default: 5000).
- `FLASK_DEBUG`: Development mode flag (1 for debug, 0 for production).

---

## 6. Key Files Directory

- `app.py`: Flask application, route definitions, ensemble scorer, heuristic rule engine, scraping functions.
- `gnn_inference.py`: PyTorch Geometric PyG graph loader, GCN model architecture, 5-NN graph augmentation, inference handler.
- `gnn_model_weighted.pt`: Saved state dictionary of trained PyG GCN model.
- `graph_data.pt`: Saved PyTorch Geometric graph data containing 40,432 nodes and 521,140 edges.
- `scaler.pkl`: Pre-fitted StandardScaler for 7-dimensional GNN feature vector normalization.
- `evaluate_model.py`: Validation script for running performance evaluation benchmarks.
- `generate_charts.py`: Visualization utility script for generating ROC curves, confusion matrices, and metrics charts.
- `templates/`: HTML Jinja2 templates for the web application UI.

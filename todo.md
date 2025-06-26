# Financial Data Analysis TODO (Indexed)

The following indexed checklist breaks the original PRD into very small, code-ready tasks. Mark each item when complete.

## 0. Environment
- [x] 0.1 Activate existing virtual environment `env_eaglefox` (Python 3.10.13)
- [x] 0.2 Add `requirements.txt` and install core libs (`dash`, `pandas`, `yfinance`, `sklearn`, `tensorflow`, `joblib`)
- [x] 0.3 Initialise Git repository & add `.gitignore` (already existed)
- [x] 0.4 Configure Ruff & pytest (with coverage) in pyproject.toml
- [x] 0.5 Add project root README (updated existing README)

## 1. Directory Skeleton
- [x] 1.1 Create folders: `/data`, `/ml`, `/frontend`, `/tests`
- [x] 1.2 Add empty `__init__.py` in every package
- [x] 1.3 Commit empty placeholder modules

## 2. Data Layer
### 2.1 Fetching (`data/fetch_data.py`)
- [ ] 2.1.1 `validate_ticker(ticker) -> bool`
- [ ] 2.1.2 `get_financials(ticker) -> DataFrame`
- [ ] 2.1.3 `get_13f_holdings(ticker) -> DataFrame`
- [ ] 2.1.4 `get_mutual_fund_holdings(ticker) -> DataFrame`
- [ ] 2.1.5 `get_corporate_actions(ticker) -> DataFrame`
- [ ] 2.1.6 `retry_api_call(func)` decorator
- [ ] 2.1.7 Unit tests for every function

### 2.2 Caching (`data/cache.py`)
- [ ] 2.2.1 `save_to_cache(key, df)`
- [ ] 2.2.2 `load_from_cache(key) -> DataFrame | None`
- [ ] 2.2.3 `is_cache_stale(key, minutes) -> bool`
- [ ] 2.2.4 Unit tests

## 3. Preprocessing (`data/preprocess.py`)
- [ ] 3.1 `impute_missing(df, method="mean")`
- [ ] 3.2 `scale_numeric(df)`
- [ ] 3.3 `encode_categorical(df)`
- [ ] 3.4 Unit tests

## 4. ML Helpers (`ml/base.py`)
- [ ] 4.1 `split_data(df, target, test_size=0.2)`
- [ ] 4.2 Metric helper functions
- [ ] 4.3 Unit tests

## 5. Classification (`ml/classification.py`)
- [ ] 5.1 Logistic Regression model
- [ ] 5.2 Random Forest Classifier
- [ ] 5.3 TensorFlow DNN Classifier
- [ ] 5.4 Metrics (accuracy, precision, recall, F1)
- [ ] 5.5 Unit tests

## 6. Regression (`ml/regression.py`)
- [ ] 6.1 Linear Regression
- [ ] 6.2 Random Forest Regressor
- [ ] 6.3 TensorFlow DNN Regressor
- [ ] 6.4 N-step prediction helper
- [ ] 6.5 Metrics (MSE, RMSE, R²)
- [ ] 6.6 Unit tests

## 7. Unsupervised (`ml/unsupervised.py`)
- [ ] 7.1 K-Means clustering
- [ ] 7.2 DBSCAN clustering
- [ ] 7.3 PCA dimensionality reduction
- [ ] 7.4 Silhouette calculation
- [ ] 7.5 Unit tests

## 8. Metric Visuals (`ml/metrics.py`)
- [ ] 8.1 Confusion matrix (Plotly)
- [ ] 8.2 Predicted vs Actual plot
- [ ] 8.3 Cluster scatter plot
- [ ] 8.4 Unit tests

## 9. Frontend Components (`frontend/components.py`)
- [ ] 9.1 Ticker & date-range form
- [ ] 9.2 Tabbed Dash `DataTable`
- [ ] 9.3 Conditional formatting rules
- [ ] 9.4 ML configuration panel
- [ ] 9.5 Results area (metrics + plots)
- [ ] 9.6 Export buttons (CSV / Excel)
- [ ] 9.7 Unit tests

## 10. Frontend Layout (`frontend/layouts.py`)
- [ ] 10.1 Build responsive layout with Bootstrap
- [ ] 10.2 Add tooltips for advanced options
- [ ] 10.3 Unit tests

## 11. Dash Callbacks (`frontend/callbacks.py`)
- [ ] 11.1 Data fetch callback
- [ ] 11.2 Table refresh callback
- [ ] 11.3 ML train callback
- [ ] 11.4 Result render callback
- [ ] 11.5 Export callback
- [ ] 11.6 Unit tests

## 12. Entrypoint (`app.py`)
- [ ] 12.1 Create Dash app instance
- [ ] 12.2 Register layouts & callbacks
- [ ] 12.3 Run Dash app locally (`python app.py`)
- [ ] 12.4 Unit tests

## 13. Utilities (`utils.py`)
- [ ] 13.1 `sanitize_input(value)`
- [ ] 13.2 Shared validation helpers
- [ ] 13.3 Unit tests

## 14. Configuration (`config.py`)
- [ ] 14.1 Define constants (cache path, retry delay, defaults)
- [ ] 14.2 Load env overrides
- [ ] 14.3 Unit tests

## 15. Test Infrastructure
- [ ] 15.1 Configure pytest fixtures
- [ ] 15.2 Create sample data fixtures
- [ ] 15.3 Mock `yfinance` responses


## 16. Documentation
- [ ] 16.1 Update README usage examples
- [ ] 16.2 Add docstrings everywhere (PEP-257)
- [ ] 16.3 Write user guide (screenshots)

---
Use the smallest possible PRs and tick tasks as you complete them. Good luck!
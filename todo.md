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
- [x] 2.1.1 `validate_ticker(ticker) -> bool`
  - [x] Unit test: Test with invalid inputs
  - [x] Unit test: Test with valid ticker
  - [x] Unit test: Test with invalid ticker
  - [x] Unit test: Test exception handling
- [x] 2.1.2 `get_financials(ticker) -> DataFrame`
  - [x] Unit test: Test with valid ticker
  - [x] Unit test: Test with invalid ticker
  - [x] Unit test: Test with empty response
  - [x] Unit test: Test exception handling
- [x] 2.1.3 `get_13f_holdings(ticker) -> DataFrame`
  - [x] Unit test: Test with valid ticker
  - [x] Unit test: Test with invalid ticker
  - [x] Unit test: Test with empty response
  - [x] Unit test: Test exception handling
- [x] 2.1.4 `get_mutual_fund_holdings(ticker) -> DataFrame`
  - [x] Unit test: Test with valid ticker
  - [x] Unit test: Test with invalid ticker
  - [x] Unit test: Test with empty response
  - [x] Unit test: Test exception handling
- [ ] 2.1.5 `get_corporate_actions(ticker) -> DataFrame`
  - [ ] Unit test: Test with valid ticker
  - [ ] Unit test: Test with invalid ticker
  - [ ] Unit test: Test with empty response
  - [ ] Unit test: Test exception handling
- [ ] 2.1.6 `retry_api_call(func)` decorator
  - [ ] Unit test: Test successful execution
  - [ ] Unit test: Test with retries needed
  - [ ] Unit test: Test max retries exceeded

### 2.2 Caching (`data/cache.py`)
- [ ] 2.2.1 `save_to_cache(key, df)`
  - [ ] Unit test: Test successful save
  - [ ] Unit test: Test with invalid inputs
  - [ ] Unit test: Test with file system errors
- [ ] 2.2.2 `load_from_cache(key) -> DataFrame | None`
  - [ ] Unit test: Test successful load
  - [ ] Unit test: Test with non-existent key
  - [ ] Unit test: Test with corrupted cache file
- [ ] 2.2.3 `is_cache_stale(key, minutes) -> bool`
  - [ ] Unit test: Test with fresh cache
  - [ ] Unit test: Test with stale cache
  - [ ] Unit test: Test with non-existent cache

## 3. Preprocessing (`data/preprocess.py`)
- [ ] 3.1 `impute_missing(df, method="mean")`
  - [ ] Unit test: Test mean imputation
  - [ ] Unit test: Test median imputation
  - [ ] Unit test: Test drop imputation
  - [ ] Unit test: Test with no missing values
- [ ] 3.2 `scale_numeric(df)`
  - [ ] Unit test: Test with numeric columns
  - [ ] Unit test: Test with mixed columns
  - [ ] Unit test: Test with empty DataFrame
- [ ] 3.3 `encode_categorical(df)`
  - [ ] Unit test: Test with categorical columns
  - [ ] Unit test: Test with mixed columns
  - [ ] Unit test: Test with empty DataFrame

## 4. ML Helpers (`ml/base.py`)
- [ ] 4.1 `split_data(df, target, test_size=0.2)`
  - [ ] Unit test: Test with valid inputs
  - [ ] Unit test: Test with different test sizes
  - [ ] Unit test: Test with stratification
- [ ] 4.2 Metric helper functions
  - [ ] Unit test: Test classification metrics
  - [ ] Unit test: Test regression metrics
  - [ ] Unit test: Test with edge cases

## 5. Classification (`ml/classification.py`)
- [ ] 5.1 Logistic Regression model
  - [ ] Unit test: Test model training
  - [ ] Unit test: Test model prediction
  - [ ] Unit test: Test model evaluation
- [ ] 5.2 Random Forest Classifier
  - [ ] Unit test: Test model training
  - [ ] Unit test: Test model prediction
  - [ ] Unit test: Test feature importance
- [ ] 5.3 TensorFlow DNN Classifier
  - [ ] Unit test: Test model architecture
  - [ ] Unit test: Test model training
  - [ ] Unit test: Test model prediction
- [ ] 5.4 Metrics (accuracy, precision, recall, F1)
  - [ ] Unit test: Test metrics calculation
  - [ ] Unit test: Test with imbalanced data

## 6. Regression (`ml/regression.py`)
- [ ] 6.1 Linear Regression model
  - [ ] Unit test: Test model training
  - [ ] Unit test: Test model prediction
  - [ ] Unit test: Test model coefficients
- [ ] 6.2 Random Forest Regressor
  - [ ] Unit test: Test model training
  - [ ] Unit test: Test model prediction
  - [ ] Unit test: Test feature importance
- [ ] 6.3 TensorFlow DNN Regressor
  - [ ] Unit test: Test model architecture
  - [ ] Unit test: Test model training
  - [ ] Unit test: Test model prediction
- [ ] 6.4 Metrics (MSE, RMSE, MAE, R²)
  - [ ] Unit test: Test metrics calculation
  - [ ] Unit test: Test with different data distributions

## 7. Unsupervised Learning (`ml/unsupervised.py`)
- [ ] 7.1 K-Means Clustering
  - [ ] Unit test: Test clustering
  - [ ] Unit test: Test optimal k selection
  - [ ] Unit test: Test with different initializations
- [ ] 7.2 PCA
  - [ ] Unit test: Test dimensionality reduction
  - [ ] Unit test: Test explained variance
  - [ ] Unit test: Test with standardized data
- [ ] 7.3 Metrics (silhouette score, explained variance)
  - [ ] Unit test: Test metrics calculation
  - [ ] Unit test: Test with different cluster shapes

## 8. Frontend Components (`frontend/components.py`)
- [ ] 8.1 Data table with conditional formatting
  - [ ] Unit test: Test table rendering
  - [ ] Unit test: Test conditional formatting
  - [ ] Unit test: Test with empty data
- [ ] 8.2 Dropdown for ticker selection
  - [ ] Unit test: Test dropdown options
  - [ ] Unit test: Test dropdown callbacks
- [ ] 8.3 Date range picker
  - [ ] Unit test: Test date selection
  - [ ] Unit test: Test date range validation
- [ ] 8.4 Download button
  - [ ] Unit test: Test download functionality
  - [ ] Unit test: Test with different file formats

## 9. Frontend Layouts (`frontend/layouts.py`)
- [ ] 9.1 Main layout
  - [ ] Unit test: Test layout rendering
  - [ ] Unit test: Test responsive design
- [ ] 9.2 Data view layout
  - [ ] Unit test: Test with different data types
  - [ ] Unit test: Test with empty data
- [ ] 9.3 ML analysis layout
  - [ ] Unit test: Test with different model outputs
  - [ ] Unit test: Test with error states

## 10. Frontend Callbacks (`frontend/callbacks.py`)
- [ ] 10.1 Update data table callback
  - [ ] Unit test: Test with valid inputs
  - [ ] Unit test: Test with invalid inputs
  - [ ] Unit test: Test error handling
- [ ] 10.2 Run ML analysis callback
  - [ ] Unit test: Test with valid model parameters
  - [ ] Unit test: Test with invalid model parameters
  - [ ] Unit test: Test with different models
- [ ] 10.3 Download data callback
  - [ ] Unit test: Test with different data formats
  - [ ] Unit test: Test with large datasets

## 11. Main App (`app.py`)
- [ ] 11.1 Initialize Dash app
  - [ ] Unit test: Test app initialization
  - [ ] Unit test: Test with different configurations
- [ ] 11.2 Register callbacks
  - [ ] Unit test: Test callback registration
  - [ ] Unit test: Test callback dependencies
- [ ] 11.3 Configure server
  - [ ] Unit test: Test server configuration
  - [ ] Unit test: Test with different environments

## 12. Utilities (`utils.py`)
- [ ] 12.1 Error handling utilities
  - [ ] Unit test: Test error capturing
  - [ ] Unit test: Test error formatting
  - [ ] Unit test: Test with different error types
- [ ] 12.2 Logging setup
  - [ ] Unit test: Test log levels
  - [ ] Unit test: Test log formatting
  - [ ] Unit test: Test log file rotation

## 13. Configuration (`config.py`)
- [ ] 13.1 Environment variables
  - [ ] Unit test: Test variable loading
  - [ ] Unit test: Test with missing variables
  - [ ] Unit test: Test with invalid values
- [ ] 13.2 App configuration
  - [ ] Unit test: Test configuration loading
  - [ ] Unit test: Test with different environments
- [ ] 13.3 Define constants (cache path, retry delay, defaults)
  - [ ] Unit test: Test constant values
  - [ ] Unit test: Test with different environments

## 14. Documentation
- [ ] 14.1 Update README with setup & usage
- [ ] 14.2 Add docstrings to all modules
- [ ] 14.3 Add inline comments for complex logic

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
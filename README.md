# Eagle1-Fox3
## Financial Data Analysis Application

Real-time concurrent data extraction + TensorFlow analysis on Yahoo Finance data.

## Overview

This Python Dash web application enables users to:

- Download financial data from Yahoo Finance (company financials, 13F/13D holdings, mutual fund holdings, corporate actions)
- View data in interactive tables with conditional formatting
- Perform machine learning analysis using scikit-learn and TensorFlow
  - Classification models
  - Regression models
  - Unsupervised learning

## Project Structure

```
/
├── data/             # Data fetching, caching, preprocessing
├── ml/               # Machine learning models and utilities
├── frontend/         # Dash UI components and callbacks
├── tests/            # Unit tests
├── app.py            # Main application entry point
├── config.py         # Configuration settings
├── utils.py          # Shared utilities
└── requirements.txt  # Dependencies
```

## Setup

1. Activate the virtual environment:
   ```
   # Windows
   .\env_eaglefox\Scripts\activate

   # Unix/MacOS
   source env_eaglefox/bin/activate
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the application:
   ```
   python app.py
   ```

## Features

- **Data Ingestion**: Fetch financial data from Yahoo Finance with caching
- **Data Visualization**: Interactive tables with conditional formatting
- **Machine Learning**: Classification, regression, and unsupervised learning
- **Export**: Download data and analysis results

## Development

- Run tests: `pytest`
- Check code quality: `ruff check .`

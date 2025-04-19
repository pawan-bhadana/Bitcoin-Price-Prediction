# Bitcoin Price Prediction

This project analyzes Bitcoin (BTC) price movements against the Indian Rupee (INR) using hourly Kline data from the Pi42 exchange API (July 2024–September 2024). Through **Exploratory Data Analysis (EDA)**, **classification models**, and **LSTM regression**, it predicts price movement directions (increase/decrease) and future price levels (next 30 hours).

## Project Goals
- Identify trends and technical indicators in BTCINR prices.
- Predict whether prices will rise or fall (classification).
- Forecast exact opening and closing prices for the next 30 hours (regression).
- Validate predictions using backtesting.

## Dataset
- **Source**: Pi42 exchange API.
- **Features**: Open, high, low, close prices, and trading volume.
- **Timeframe**: July 2024–September 2024, hourly intervals.

## Repository Contents
- `EDA.ipynb`: Analyzes price distributions, correlations, outliers, volume trends, and moving averages.
- `DataDivision.ipynb`: Preprocesses and splits data for modeling.
- `ML_Classificaion.ipynb`: RandomForest and XGBoost models to predict price movement direction.
- `close_LSTM_regression.ipynb`: LSTM model to forecast closing prices (next 30 hours).
- `open_LSTM.ipynb`: LSTM model to forecast opening prices.
- `Pi42_open1h.ipynb`: Additional 1-hour opening price analysis.
- `Main_Report.docx`: Comprehensive project report.
- `.gitignore`: Excludes temporary files (e.g., `.ipynb_checkpoints`, `__pycache__`).

## Key Findings
### Exploratory Data Analysis (EDA)
- **Data Quality**: No null values.
- **Distributions**: Price variables (open, high, low, close) follow a Gaussian distribution (visualized with histplots).
- **Correlations**: High correlation among price variables; only one used for predictions to avoid redundancy.
- **Outliers**: Capped at 0.05 and 0.95 quantiles to reduce impact, preserving time-series integrity.
- **Price Trends**: Significant price drop in early August 2024, with recovery lagging until September.
- **Volume**: Spike in trading volume post-August drop (around 7–8 August).
- **Moving Averages**: Computed short-term (2, 7, 15 hours) and long-term (50 hours) SMAs to identify trends.
- **Weekly Trends**: Opening prices increased as weeks progressed.

### Modeling
#### Classification (Price Movement)
- **Models**: RandomForestClassifier, XGBoostClassifier.
- **Preprocessing**: StandardScaler (Gaussian distribution).
- **Hyperparameter Tuning**: GridSearchCV for `n_estimators`, `max_depth`, etc.
- **Evaluation**: Precision score with backtesting (e.g., 0.605 for low price).

#### Regression (Price Forecasting)
- **Model**: LSTM (Long Short-Term Memory).
- **Approach**: Used last 100 hours of prices to predict the 101st hour (open/close).
- **Preprocessing**: StandardScaler.
- **Evaluation**: Precision score with backtesting.

## Future Improvements
- Add external features (e.g., market sentiment, social media trends).
- Explore advanced models (e.g., Transformer-based architectures).
- Extend backtesting for longer periods.

## Requirements
Install the following Python libraries to run the notebooks:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost tensorflow

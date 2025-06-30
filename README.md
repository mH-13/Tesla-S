# Tesla‑S: Tesla Stock Price Forecasting

## Project Overview
Financial time‑series forecasting on Tesla’s historical stock data. We’ll build and compare baseline, traditional, machine‑learning, and deep‑learning models to predict next‑month closing prices.

## Folder Structure
```text
Tesla-S/
├── data/
│   ├── raw/              # Source CSVs (unchanged)
│   └── processed/        # Cleaned and feature‑engineered data
├── notebooks/            # Jupyter notebooks for EDA & experiments
├── src/                  # Production scripts and modules
│   ├── data/             # Data ingestion & cleaning
│   ├── features/         # Feature engineering
│   ├── models/           # Model training & inference
│   └── utils/            # Helper functions (plots, logging, etc.)
├── tests/                # Unit tests
├── requirements.txt      # Python dependencies
└── README.md             # Project introduction and instructions

```

## Setup Instructions

1. Clone the repo:  
   ```bash
   git clone https://github.com/mH-13/Tesla-S
   cd Tesla‑S
````

2. Create and activate a virtual environment:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Place the CSV in `data/raw/` as `TSLA.csv`.



## Further Implementation:

1. **Data ingestion & cleaning** (`src/data/make_dataset.py`)
2. **Feature engineering** (`src/features/build_features.py`)
3. **EDA notebook** (`notebooks/01_eda.ipynb`)
4. **Model training & evaluation** (`src/models/train.py`)
5. **Prediction & visualization** (`src/models/predict.py`)
6. **Unit tests** (`tests/`)

````
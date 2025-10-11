# Home Credit Feature Engineering Pipeline

This repository contains a lightweight feature-engineering pipeline for the [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk) Kaggle competition.  
The goal is to aggregate the raw relational tables provided by the competition into a single modeling table (`combined.csv`) that can be used by downstream modelling experiments.

## Repository Layout

```
.
├── combine.py                 # Script that stitches base applications with engineered feature tables
├── data_transformation/       # Jupyter notebooks that produce per-table aggregations
├── transformed_data/          # (Generated) CSV feature tables keyed by SK_ID_CURR
├── home-credit-default-risk/  # (User-provided) raw Kaggle dataset CSVs
├── cluster.ipynb              # Optional clustering exploration notebook
└── test.ipynb                 # Sandbox notebook for ad-hoc experiments
```

> **Note**: `transformed_data/` and `home-credit-default-risk/` are ignored by Git. They are created locally when you run the notebooks and download the Kaggle data respectively.

## Prerequisites

- Python 3.8 or newer
- [pip](https://pip.pypa.io/en/stable/)
- [pandas](https://pandas.pydata.org/) (required by `combine.py`)
- Optional: Jupyter Notebook / JupyterLab for running the feature engineering notebooks

You can install the Python dependency with:

```bash
pip install pandas
```

## Getting the Raw Data

1. Download the Home Credit Default Risk dataset from Kaggle (`application_train.csv`, `application_test.csv`, and the related bureau/credit-card/installments/etc. tables).
2. Create a folder named `home-credit-default-risk/` in the repository root.
3. Place all of the downloaded CSVs into that directory.

The repository assumes that the following files are available:

- `home-credit-default-risk/application_train.csv`
- `home-credit-default-risk/application_test.csv`
- Additional feature tables such as `bureau.csv`, `bureau_balance.csv`, `credit_card_balance.csv`, `installments_payments.csv`, `POS_CASH_balance.csv`, and `previous_application.csv`

## Workflow

1. **Run the feature notebooks** in `data_transformation/`. Each notebook loads one of the raw tables, engineers features, and writes a CSV to `transformed_data/` using `SK_ID_CURR` as the key.  
   Typical outputs include:
   - `_bureau.csv`
   - `_bureau_balance.csv`
   - `_credit_card_balance.csv`
   - `_installments.csv`
   - `_pos_cash.csv`
   - `_prev_features.csv`

2. **Combine the features** by executing the merge script:

   ```bash
   python combine.py
   ```

   The script will:
   - Load the base application train and test tables.
   - Drop the `TARGET` column from the training data to retain only features.
   - Coerce `SK_ID_CURR` to `int64` and align train/test schemas.
   - Concatenate the train and test rows into one base table.
   - Discover every `*.csv` file in `transformed_data/`, clean stray index columns, de-duplicate keys, and left-join on `SK_ID_CURR`.
   - Suffix colliding feature names deterministically using the source file stem.
   - Export the final merged dataset to `combined.csv` in the project root.

3. **Inspect or model** using the resulting `combined.csv` dataset.

## Troubleshooting

- **Missing files** – Ensure all Kaggle CSVs are downloaded into `home-credit-default-risk/` and that the notebooks have written their outputs to `transformed_data/`.
- **Key casting errors** – If a source file contains non-numeric IDs, they will be dropped during coercion to integers. Inspect the offending CSV for malformed `SK_ID_CURR` values.
- **Column name collisions** – When two tables produce the same feature name, the column from the joined table is suffixed with the file stem (for example `_credit_card_balance`). Rename the column in the notebook if you prefer a custom suffix.

## Extending the Pipeline

- Create a new notebook (or script) that produces features keyed by `SK_ID_CURR` and saves the result as a CSV into `transformed_data/`.
- Re-run `python combine.py`; the script will automatically pick up the new file and incorporate the features.
- To add automated tests or integrate the pipeline into a larger project, wrap `combine.py` in your preferred workflow orchestration tool or call its functions directly from Python.

## License

This project does not currently include an explicit license. If you plan to share or redistribute your modifications, please add one that matches your needs.

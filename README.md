# Craigslist Used Vehicle Asking Prices

[简体中文](README.zh-CN.md)

SDSC 5002 team project. We use exploratory data analysis to study how advertised used vehicle prices vary with age, odometer reading, vehicle attributes, and state. The dataset records asking prices for Craigslist listings; it does not measure completed sales or establish causal effects.

[Project proposal](docs/proposal.md) · [Data source and setup](DATA.md)

## Run the notebooks

Use Python 3.11 or newer. From the repository root, create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
# macOS/Linux: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Follow [DATA.md](DATA.md) to get the versioned Kaggle file at `data/raw/vehicles.csv`. Start Jupyter with `python -m jupyterlab` and run the notebooks in order:

| Notebook | What it does |
| --- | --- |
| [01 · Import and profile](notebooks/01_import_and_profile.ipynb) | Checks the file, schema, missingness, duplicate IDs, and numeric summaries. |
| [02 · Clean and derive](notebooks/02_clean_and_derive.ipynb) | Audits how cleaning changes row counts and price, derives vehicle age, and writes `data/processed/listings.csv` locally. |
| [03 · Price, age, and mileage](notebooks/03_price_age_mileage.ipynb) | Plots price distributions and relationships, checks extreme prices, and reports correlations with sensitivity checks. |
| [04 · Categories and states](notebooks/04_categories_and_states.ipynb) | Compares prices across condition, fuel, vehicle type, and manufacturer, and ranks state medians with sample sizes. |

The notebooks retain positive prices, flagging very low or high prices for review. Missing fields are handled separately for each question; each plot or summary reports the relevant sample size. See the cleaning and analysis notebooks for the exact rules and their effect.

## Work together

Use issues to track tasks and questions. Create a branch for each change and open a pull request with a short summary, the notebooks you reran, and any changes to filters or results. Coordinate before editing the same notebook to avoid conflicts. Keep shared transformations in `src/`, with notebook code focused on the analysis. Keep only reviewed aggregate figures and tables in notebook outputs; do not commit source or processed data, listing descriptions, identifiers, or credentials.

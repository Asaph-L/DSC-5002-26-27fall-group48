# Data source and setup

[简体中文](DATA.zh-CN.md)

This project uses Austin Reese's [Used Cars Dataset (Craigslist Cars & Trucks Data)](https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data) on Kaggle. Use [version 10](https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data/versions/10), released on 6 May 2021 with posting dates. The project pins that snapshot because the proposal describes roughly 426,880 listings and 26 columns; newer versions may have a different population. The recorded price is a seller's advertised asking price in USD, and a listing does not establish a sale.

## Get the file

Download version 10 from the link above. If Kaggle prompts for sign-in, use your own Kaggle account. Extract `vehicles.csv` to `data/raw/vehicles.csv` under the repository root. Keep the original bytes; do not open and save it in a spreadsheet before running the notebooks.

Alternatively, from the repository root, use the official [KaggleHub client](https://github.com/Kaggle/kagglehub#download-dataset):

```bash
python -m pip install kagglehub
python -c "import kagglehub; kagglehub.dataset_download('austinreese/craigslist-carstrucks-data/versions/10', path='vehicles.csv', output_dir='data/raw')"
```

## Verify the snapshot

The original version 10 archive contains 426,880 listing records and 26 columns in `vehicles.csv` (1,447,955,215 bytes). The SHA-256 of the extracted CSV is `54c60e20aa494a73f7f9170bff1e38b4800472b767cab8f9dab7da5dcc261a5c`.

To compare your copy with that fingerprint:

```bash
python -c "import hashlib; f=open('data/raw/vehicles.csv','rb'); print(hashlib.file_digest(f, 'sha256').hexdigest())"
```

Use [01 · Import and profile](notebooks/01_import_and_profile.ipynb) to check the header and row count before cleaning. Keep the version and file fingerprint with any results you share.

## Group 48 cleaned ZIP

The 4.74 MB archive contains the original `listings.csv` produced by notebook 02, with 393,985 rows and 18 columns. Extract it to `data/processed/listings.csv` to run notebooks 03 and 04. The archive includes bilingual setup instructions and a checksum. Notebooks that read the original Kaggle snapshot still require `data/raw/vehicles.csv`. [Download the cleaned ZIP](downloads/DSC5002_Group48_cleaned_listings.zip)

## What belongs in GitHub

Kaggle lists the dataset as **CC0: Public Domain**. The original listings may include seller-entered descriptions, vehicle identifiers, precise locations and links. [Craigslist's terms](https://www.craigslist.org/about/terms.of.use/en) restrict redistribution of posting content; the CSV also exceeds [GitHub's ordinary 100 MiB file limit](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github). Team members obtain the same raw snapshot from Kaggle and store it in the ignored local `data/` folder.

The approved cleaned ZIP above is the only public row-level snapshot for this project. It contains structured analysis fields such as asking price, model year, mileage and state. Listing IDs, URLs, descriptions, VINs and coordinates are excluded from the archive. Keep other locally generated data and caches in ignored directories. Public notebook outputs use aggregate charts and tables.

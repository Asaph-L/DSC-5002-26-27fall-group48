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

## What belongs in GitHub

Kaggle lists the dataset as **CC0: Public Domain**. The original listings may include seller-entered descriptions, vehicle identifiers, precise locations and links. [Craigslist's terms](https://www.craigslist.org/about/terms.of.use/en) restrict redistribution of posting content; the CSV also exceeds [GitHub's ordinary 100 MiB file limit](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github). Team members obtain the same snapshot from Kaggle. Keep raw and processed listings in the local `data/` folder, which Git ignores, and share code plus reviewed aggregate charts and tables. Do not publish individual listing records, URLs, descriptions, VINs, or coordinates in notebook outputs.

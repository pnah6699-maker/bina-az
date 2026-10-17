# Data

The **bina.az sale** dataset is **not committed** to the repo (see `.gitignore`).
Each team member downloads it locally.

## Where to get it
Kaggle: **"binaaz-sale-project"** by `sehriyarmemmedli`
<https://www.kaggle.com/datasets/sehriyarmemmedli/binaaz-sale-project>

## How to place it
Either run the helper (uses `kagglehub`, no API key needed for this public dataset):

```bash
python scripts/download_data.py
```

or download the archive by hand and extract it so the CSV sits here:

```
data/house_sale.csv
```

The path is set once in `src/data_prep.py` (`DATA_PATH`); you can also
override it with the environment variable `BINAAZ_DATA=/path/to/file.csv`.

## Version used for the report
- Kaggle dataset version **1**, file `house_sale.csv` (141,882,994 bytes,
  100,775 rows × 51 columns), downloaded 2026-10-04.
- Listings scraped 2024-10-05 … 2024-10-16; every price is in AZN.

## Notes / gotchas
- Column names are partly **Azerbaijani** (`Sahə`, `Otaq sayı`, `Mərtəbə`,
  `Təmir`, `Kateqoriya`, …). Read with UTF-8.
- **Leakage:** `unit_price` (= price / area) and `total_price` (= price) are
  dropped before anything else happens; see `LEAKAGE_COLUMNS` in
  `src/data_prep.py` and §III of the report.
- The same listing appears up to several times (scraped on different days):
  `clean()` keeps the latest scrape per item URL.

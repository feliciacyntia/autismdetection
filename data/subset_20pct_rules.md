# subset_20pct.csv — aturan pembuatan

- Sumber: `folds.csv` (sha256 48f8e8bf9b8df9b7ce96808dda4f2c450c084ac7de22fad46ff89cfe0fe285b8), `pilot_ids.csv` (sha256 44ed207104816fa251d48d113d26214983cba8757aa8cb48d11cc505a5f60433)
- Hasil: `subset_20pct.csv` (sha256 da48e17a11ad2a63b81fa997f364d3715a5da1e8955cb4b021da3e71e61b24b0), 200 subjek, semua `split == dev`, seed 42
- Kandidat hanya dev; test held-out tidak pernah masuk.
- Alokasi dua tingkat (largest remainder, seri diputus acak seed 42): kuota SITE_ID × label proporsional terhadap dev, lalu dibagi ke fold_strat sebanding jumlah subjek per fold.
- Semua 20 subjek pilot wajib masuk (kuota strata minimal = jumlah pilot di strata).
- Di dalam strata: pilot dulu, sisanya acak seed 42.
- File tidak memuat label; label tetap hanya di folds.csv.

## Proporsi ASD: dev 0.4666 | subset 0.4750

## Per fold

| fold_strat | dev_n | subset_n | dev_asd | subset_asd |
|---|---|---|---|---|
| 0 | 171 | 41 | 0.45 | 0.488 |
| 1 | 171 | 38 | 0.45 | 0.421 |
| 2 | 171 | 41 | 0.48 | 0.488 |
| 3 | 170 | 43 | 0.465 | 0.465 |
| 4 | 170 | 37 | 0.488 | 0.514 |

## Per situs

| SITE_ID | dev_n | subset_n | expected | dev_asd | subset_asd |
|---|---|---|---|---|---|
| CALTECH | 32 | 8 | 7.5 | 0.5 | 0.5 |
| CMU | 23 | 5 | 5.4 | 0.522 | 0.6 |
| KKI | 37 | 8 | 8.7 | 0.378 | 0.375 |
| LEUVEN_1 | 25 | 6 | 5.9 | 0.48 | 0.5 |
| LEUVEN_2 | 30 | 7 | 7.0 | 0.433 | 0.429 |
| MAX_MUN | 44 | 10 | 10.3 | 0.386 | 0.4 |
| NYU | 155 | 37 | 36.3 | 0.432 | 0.432 |
| OHSU | 24 | 6 | 5.6 | 0.458 | 0.5 |
| OLIN | 25 | 6 | 5.9 | 0.48 | 0.5 |
| PITT | 48 | 11 | 11.3 | 0.521 | 0.545 |
| SBL | 26 | 6 | 6.1 | 0.5 | 0.5 |
| SDSU | 24 | 6 | 5.6 | 0.292 | 0.333 |
| STANFORD | 23 | 5 | 5.4 | 0.522 | 0.6 |
| TRINITY | 39 | 9 | 9.1 | 0.462 | 0.444 |
| UCLA_1 | 55 | 13 | 12.9 | 0.564 | 0.538 |
| UCLA_2 | 17 | 4 | 4.0 | 0.529 | 0.5 |
| UM_1 | 76 | 18 | 17.8 | 0.395 | 0.389 |
| UM_2 | 29 | 7 | 6.8 | 0.379 | 0.429 |
| USM | 76 | 18 | 17.8 | 0.605 | 0.611 |
| YALE | 45 | 10 | 10.6 | 0.489 | 0.5 |

# Run uji (bukan hasil)

Run singkat untuk menguji runner sebelum run sebenarnya: smoke test, uji resume/STOP, uji laporan, uji v3 (1 fold, 1 seed, 3 trial).
Disimpan sebagai catatan bahwa runner diuji (semua percobaan dicatat). **Angka di sini tidak boleh dipakai sebagai hasil.**

| folder | isi |
|---|---|
| `_smoke`, `_smoke2` | smoke test runner pilot v1 (epoch sangat sedikit) |
| `_v2smoke2` | smoke test runner pilot v2 |
| `_resume_test` | uji berhenti (STOP) lalu `--resume` |
| `_report_test` | uji pembuatan tabel/laporan |
| `v3_test` | uji runner v3 tabel: fold 0, seed 0, 3 trial Optuna, LOSO + uji STOP/resume |

Checkpoint `.pt` dan cache embedding `.npy` run uji tidak disertakan.

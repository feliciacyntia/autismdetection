# Rekomendasi Arsitektur dan Preprocessing sMRI + Fenotipik ABIDE I

## Kesimpulan singkat

Bukti penggunaan **ViT untuk klasifikasi ASD berbasis sMRI ABIDE multi-site** masih terbatas. Beberapa paper melaporkan performa tinggi, tetapi protokol validasi atau detail metodenya perlu diperiksa secara kritis. Untuk tugas akhir dengan sekitar 1.000 subjek, kandidat yang relatif realistis adalah **2D multi-view dengan backbone pretrained ringan + MLP fenotipik**. **CNN 3D kecil** dapat menjadi alternatif, sedangkan ViT/Swin lebih cocok sebagai pembanding jika sumber daya komputasi memungkinkan.

> Catatan: status indeks Scopus dari paper-paper di bawah belum diverifikasi. Periksa langsung melalui Scopus Sources atau SCImago sebelum menyebut status indeks di laporan.

## 1. Paper yang ditemukan

| Paper | Data dan metode | Hasil yang dilaporkan | Catatan kritis / keterbatasan |
|---|---|---|---|
| Gao & Xu, *Japanese Journal of Radiology* (2025), 43(11):1788–1802. [DOI](https://doi.org/10.1007/s11604-025-01832-3) | sMRI T1; 79 ASD dan 105 kontrol; gabungan CNA (KNN)–ViT (NN); validasi silang 5-fold | Akurasi 0,951 ± 0,067; AUC 0,980 ± 0,020 | Jumlah subjek 184 sama dengan jumlah yang tercantum untuk situs NYU pada studi lain. Abstrak hanya menyebut ABIDE, sehingga kemungkinan data berasal dari satu situs; ini inferensi, bukan klaim paper. Validasi eksternal belum terlihat. |
| Polavarapu dkk., *Journal of Sensor and Actuator Networks* (2026), 15(1):21. [DOI](https://doi.org/10.3390/jsan15010021) | ABIDE I; sekitar 1.100 sampel dari 17 situs; satu irisan axial 224×224; ResNet-50; fusi awal dan ensemble; fitur fenotipik mencakup usia, jenis kelamin, FIQ, status mata, dan QA | Akurasi fusi sekitar 0,97; AUC 0,972; sMRI saja sekitar 0,87–0,88 | Perlu kehati-hatian: status mata dan anotasi QA berpotensi menjadi proksi situs; split 80/10/10 tidak tampak selaras dengan confusion matrix berjumlah 220; hasil leave-one-site-out tidak tampak pada bagian yang ditinjau; abstrak menyebut volumetri/ketebalan kortikal, tetapi metode yang dijelaskan memakai satu irisan. Jangan jadikan angka sebagai target performa tanpa pemeriksaan lebih lanjut. |
| Garcia & Kelly, *PLOS ONE* (2024). [Artikel](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0276832) | ABIDE I+II, n = 1.329; CNN 3D pada sMRI; preprocessing minimal di ruang native, tanpa normalisasi ke template | Disebut setara dengan studi machine learning sebelumnya; angka akurasi/AUC tidak berhasil diverifikasi | Referensi relevan untuk pendekatan 3D dan dataset yang lebih besar, tetapi angka performanya belum dapat dipastikan dari sumber yang ditinjau. |
| Gao dkk., *Brain Sciences* (2023), 13(11):1578. [Artikel](https://www.mdpi.com/2076-3425/13/11/1578) | Membandingkan ConvNeXt, MobileNet, Swin, dan ViT pada irisan sMRI; data hanya dari situs NYU ABIDE | Angka tidak berhasil diekstrak | Cakupan situs terbatas; hasil tidak otomatis mewakili generalisasi multi-site. |
| Mishra & Pati, PF-ViTs (2026). [DOI](https://doi.org/10.1177/1088467X251330464) | ViT-B/16 dan ViT-B/32 untuk deteksi ASD dari sMRI | Tidak terverifikasi | Nama jurnal, angka performa, dan protokol validasi belum dapat dipastikan dari halaman yang ditinjau. |
| DAGMNet, *Biomedicines* (2025), 13(9):2168. [DOI](https://doi.org/10.3390/biomedicines13092168) | sMRI + fMRI + fenotipik dengan graf populasi pada ABIDE I | Tidak dicantumkan di sini | Pembanding multimodal, tetapi melibatkan fMRI sehingga tidak setara dengan eksperimen sMRI + fenotipik saja. |
| Hierarchical Multi-View. [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC13027856/) | Proyeksi axial, sagittal, dan coronal; hanya situs NYU | Tidak terverifikasi | Nama jurnal dan DOI belum diverifikasi. |
| Preprint arXiv 2603.15168. [PDF](https://arxiv.org/pdf/2603.15168) | Graf sMRI + fMRI; evaluasi LOSO pada 17 situs | Rata-rata LOSO 82,0% dilaporkan | Masih preprint, belum peer-reviewed; juga melibatkan fMRI. |
| Tinjauan sistematis, *Applied Sciences* (2025), 15(19):10551. [DOI](https://doi.org/10.3390/app151910551) | Tinjauan penelitian klasifikasi ASD berbasis neuroimaging | Menyoroti terbatasnya validasi eksternal dan harmonisasi antar-situs | Dapat mendukung alasan pentingnya evaluasi generalisasi lintas situs. |

**Batas bukti:** paper-paper yang ditinjau tidak menyediakan angka parameter atau FLOPs yang dapat diverifikasi di sini. Sebagian studi ViT menggunakan satu situs atau irisan 2D, sehingga performanya tidak langsung dapat dibandingkan dengan dataset sekitar 1.004 subjek dari banyak situs.

## 2. Perbandingan kandidat arsitektur

| Kandidat | Bukti relevan | Kebutuhan komputasi (perkiraan kualitatif) | Kelebihan | Risiko / keterbatasan |
|---|---|---|---|---|
| **2D multi-view + CNN pretrained + MLP fenotipik** | Studi sMRI 2D; studi multi-view pada NYU; paper multimodal 2026 yang perlu ditinjau kritis | Rendah–sedang; relatif cocok untuk GPU Colab | Praktis, mudah menggabungkan fenotipik, eksperimen ablation lebih mudah | Kehilangan sebagian konteks 3D; harus memastikan pembagian data dilakukan per subjek, bukan per irisan |
| **CNN 3D kecil** | Garcia & Kelly (2024), ABIDE I+II | Sedang–tinggi; membutuhkan VRAM lebih besar | Memanfaatkan konteks spasial 3D secara langsung | Lebih berat dilatih; ukuran input dan arsitektur harus diuji lewat pilot |
| **ViT/Swin pretrained 2D** | Gao & Xu (2025), *Brain Sciences* (2023), PF-ViTs (detail belum terverifikasi) | Sedang–tinggi, tergantung ukuran model | Selaras dengan minat pada arsitektur transformer modern | Bukti sMRI multi-site masih terbatas; model besar berisiko overfit pada sekitar 1.000 subjek |
| **Fitur morfometri + model klasik** | Banyak digunakan dalam penelitian sMRI; perlu pencarian paper spesifik sebelum dipilih | Model ringan, tetapi preprocessing morfometri dapat memakan waktu | Baseline yang kuat dan relatif mudah dijelaskan | Membutuhkan pipeline ekstraksi fitur yang konsisten |
| **GNN populasi** | DAGMNet dan preprint terkait | Sedang–tinggi | Dapat memodelkan hubungan antarsubjek | Tidak cocok sebagai kandidat utama bila hanya memakai sMRI + fenotipik tanpa fMRI atau konstruksi graf yang jelas |

### Rekomendasi kandidat

1. **Kandidat utama untuk diuji:** 2D tri-planar/multi-view + backbone pretrained yang ringan + MLP fenotipik, dengan fusi fitur (*concatenation*) pada tahap menengah.
2. **Alternatif:** CNN 3D kecil dengan input volume yang diperkecil.
3. **Pembanding opsional:** ViT-S atau Swin-T pretrained jika komputasi memadai. Sebaiknya menjadi pembanding terlebih dahulu, bukan pilihan utama hanya karena lebih baru.

Backbone akhir belum perlu dikunci sebelum uji pilot dan peninjauan kesesuaian paper. Perhatikan bahwa EfficientNet-B0 dan ResNet-18 adalah kandidat yang disebut dalam rekomendasi awal, bukan keputusan final.

### Baseline yang disarankan

| Baseline | Input | Tujuan |
|---|---|---|
| Site-only | `SITE_ID` | Memeriksa apakah model dapat memprediksi label hanya dari pola antar-situs |
| Fenotipik-only | `AGE_AT_SCAN`, `SEX`, `FIQ`, `FIQ_missing` | Mengukur kontribusi data fenotipik tanpa citra |
| sMRI-only | Citra sMRI | Mengukur kontribusi citra tanpa fenotipik |
| Multimodal | sMRI + fenotipik | Mengukur apakah penggabungan modalitas memberi peningkatan yang konsisten |

Baseline site-only penting karena proporsi ASD/TD dapat berbeda antar-situs. `SITE_ID` sebaiknya digunakan untuk audit, stratifikasi, atau evaluasi lintas situs—bukan otomatis dijadikan fitur utama model klinis.

## 3. Pilihan preprocessing berdasarkan kandidat

Tabel berikut adalah **rekomendasi metodologis**, bukan klaim bahwa semua paper yang dirujuk menggunakan setiap langkah ini. Pipeline final perlu disesuaikan dengan model dan kondisi citra.

| Langkah | 2D multi-view | CNN 3D kecil | Status / catatan |
|---|---|---|---|
| Pemeriksaan file, dimensi, affine, voxel spacing, dan nilai nonfinite | Ya | Ya | Wajib sebagai pemeriksaan teknis awal |
| Reorientasi ke orientasi konsisten (misalnya RAS) | Ya | Ya | Disarankan; verifikasi orientasi dan affine, bukan sekadar mengubah label orientasi |
| Bias-field correction (misalnya N4) | Uji manfaatnya | Uji manfaatnya | Umumnya bermanfaat, tetapi bandingkan sebelum/sesudah dan jangan anggap wajib tanpa evaluasi |
| Skull stripping | Biasanya berguna | Biasanya berguna | Periksa mask secara visual; kegagalan ekstraksi otak bisa merusak fitur |
| Registrasi afin ke MNI152 | Sering membantu untuk irisan multi-view yang konsisten | Opsional, tergantung desain | Jika memakai koordinat irisan yang sama antar-subjek, standardisasi ruang menjadi penting |
| Registrasi non-linear | Opsional | Opsional | Tidak perlu dimasukkan di pilot awal kecuali ada alasan metodologis yang jelas |
| Resampling ke voxel spacing konsisten | Ya | Ya | Pilih spacing berdasarkan resolusi sumber dan kebutuhan model; hindari mengklaim 1–2 mm selalu wajib |
| Normalisasi intensitas | Ya | Ya | Contoh: z-score di dalam mask otak atau normalisasi robust; fit parameter hanya pada data latih bila relevan |
| Crop/pad ke ukuran input tetap | Ya | Ya | Ukuran akhir mengikuti arsitektur dan harus mempertahankan area otak yang relevan |
| Pemilihan beberapa irisan axial/sagittal/coronal | Ya | Tidak diperlukan | Untuk multi-view, hindari hanya mengandalkan satu irisan tengah tanpa justifikasi |
| Harmonisasi antar-situs (mis. ComBat) | Opsional | Opsional | Perlu alasan dan validasi hati-hati; estimasi parameter hanya dari data latih untuk mencegah leakage |

**Pencegahan data leakage:** imputasi FIQ, standardisasi fitur fenotipik, harmonisasi yang mempelajari parameter, dan pemilihan ambang harus di-*fit* hanya pada bagian data latih di setiap fold. Untuk pendekatan multi-view, semua irisan dari satu subjek harus tetap berada di fold yang sama dan prediksi digabungkan pada tingkat subjek.

## 4. Rencana pilot sebelum preprocessing penuh

| Tahap | Kegiatan | Output yang diharapkan |
|---|---|---|
| 1. Audit dataset | Validasi `smri_path`, ID unik, label, `SITE_ID`, dimensi, affine, spacing, dan status QC | Daftar subjek valid dan catatan masalah |
| 2. Tinjau QC | Periksa alasan 98 subjek EXCLUDE dan sampel citra dari kelompok PASS/REVIEW | Kriteria retensi yang dapat dijelaskan dan tidak semata-mata berdasarkan label/situs |
| 3. Tetapkan evaluasi | Buat split per subjek; pertimbangkan stratifikasi label dan situs serta evaluasi leave-one-site-out pada situs yang memadai | File pembagian data yang dibekukan sebelum modeling |
| 4. Pilot preprocessing | Pilih sekitar 20–30 subjek yang mencakup beberapa situs dan status QC; uji reorientasi, resampling, skull stripping, dan normalisasi | Citra sebelum/sesudah, visualisasi mask, waktu proses, log kegagalan |
| 5. Baseline fenotipik/site | Jalankan site-only dan fenotipik-only dengan pipeline yang bebas leakage | Titik pembanding sebelum model citra |
| 6. Bandingkan kandidat | Uji 2D multi-view dan CNN 3D kecil pada subset beberapa situs; catat metrik, waktu, VRAM, dan parameter | Bukti praktis untuk memilih arsitektur |
| 7. Eksperimen penuh | Jalankan model terpilih, ablation sMRI-only vs fenotipik-only vs multimodal, beberapa seed, dan analisis ketidakpastian | Hasil utama yang lebih dapat dipertanggungjawabkan |

## 5. Hal yang perlu diperiksa dari Notebook 01–04

| Temuan saat ini | Tindak lanjut |
|---|---|
| 98 subjek EXCLUDE; 411 REVIEW | Periksa apakah aturan QC memang sesuai untuk sMRI T1-weighted. Bandingkan distribusi label dan situs pada kelompok yang dikeluarkan. Jangan menyimpulkan semua REVIEW buruk tanpa inspeksi citra. |
| Dataset mencatat 20 situs, sedangkan ABIDE I umumnya dilaporkan memiliki 17 situs | Audit ejaan/varian `SITE_ID` dan dokumentasikan pemetaan situs. Hal ini memengaruhi stratifikasi dan evaluasi lintas situs. |
| `FIQ_missing` untuk sebagian subjek dan missingness dapat bervariasi menurut situs | Audit korelasi dengan situs; pertahankan indikator missing bila FIQ digunakan, tetapi pastikan tidak menjadi proksi situs yang tidak disengaja. |
| Notebook 01–04 telah dilakukan | Tidak perlu mengulang semuanya dari awal. Perbaiki atau jalankan ulang bagian yang terdampak jika audit QC, label, atau pemetaan situs menemukan masalah. Tambahkan tahap pembekuan split sebelum preprocessing/modeling. |

## 6. Sumber

1. Gao & Xu (2025), *Japanese Journal of Radiology*: https://link.springer.com/article/10.1007/s11604-025-01832-3
2. Polavarapu dkk. (2026), *Journal of Sensor and Actuator Networks*: https://doi.org/10.3390/jsan15010021
3. Garcia & Kelly (2024), *PLOS ONE*: https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0276832
4. *Brain Sciences* (2023), 13(11):1578: https://www.mdpi.com/2076-3425/13/11/1578
5. Mishra & Pati, PF-ViTs: https://doi.org/10.1177/1088467X251330464
6. DAGMNet, *Biomedicines* (2025): https://doi.org/10.3390/biomedicines13092168
7. Hierarchical Multi-View: https://pmc.ncbi.nlm.nih.gov/articles/PMC13027856/
8. Preprint arXiv 2603.15168: https://arxiv.org/pdf/2603.15168
9. Tinjauan sistematis, *Applied Sciences* (2025): https://doi.org/10.3390/app151910551
10. Hybrid Deep Learning Approaches (HD-BET): https://www.researchgate.net/publication/396541107_Hybrid_Deep_Learning_Approaches_for_Classifying_Autism_from_Brain_MRI

---

**Catatan penggunaan:** dokumen ini merapikan ringkasan riset yang diberikan menjadi Markdown dengan tabel. Detail yang belum terverifikasi tetap ditandai demikian; verifikasi sumber primer dan status indeks jurnal sebelum digunakan sebagai klaim dalam proposal atau laporan akhir.

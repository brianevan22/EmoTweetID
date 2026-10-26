# EmoTweetID: pipeline analisis + dashboard

Klasifikasi emosi dan pemodelan topik pada 3,1 juta tweet berbahasa Indonesia (IndoBERTweet + BERTopic).
Semua angka di dashboard dihitung dari CSV Anda; tidak ada angka yang ditulis manual.

## STATUS hasil yang disertakan (baca dulu)
Folder `results/` sudah berisi **hasil asli awal** dari data EmoTweetID v5 Anda, dihitung di sandbox tanpa GPU:
audit penuh 3.126.987 tweet, model klasik (SVM/NB/LogReg) dengan 5-fold CV, uji H2, diagnostik validitas,
dan pelabelan **sampel 100.000 tweet** korpus memakai LogReg. **Belum ada**: IndoBERTweet (H1), pelabelan
seluruh korpus, dan BERTopic (H3). Ketiganya dijalankan di Kaggle/Colab GPU sesuai langkah di bawah dan akan
menimpa isi `results/`.

## Struktur
| File | Fungsi | Keluaran (folder `results/`) |
|---|---|---|
| `01_audit.py` | Audit data: duplikat, tumpang tindih, panjang, uji keterwakilan | `audit.json` |
| `02_train_eval.py` | SVM / NB / LogReg (TF-IDF) vs IndoBERTweet, stratified 5-fold CV; **uji H1** (McNemar), **uji H2** (κ + bootstrap) | `metrics.json`, `oof_predictions.csv` |
| `03_label_corpus.py` | Latih ulang model terbaik, label seluruh korpus, ambang keyakinan, kata khas per emosi | `corpus_labeled.parquet`, `corpus_summary.json`, `corpus_sample.csv` |
| `04_topics.py` | BERTopic pada sampel berstrata; **uji H3** (χ², Cramér's V, residual) | `topics.json`, `h3.json`, `topic_*.csv` |
| `05_diagnostics.py` | Diagnostik: ketergantungan pada kata emosi eksplisit, pergeseran domain korpus | `diagnostics.json` |
| `app.py` | Dashboard Streamlit | – |

## Langkah (Kaggle Notebook, GPU T4/P100, internet ON)
1. Unggah tiga CSV (`Corpus`, `Human`, `Lexicon`; abaikan file terjemahan Inggris) sebagai Kaggle Dataset. Unggah folder ini sebagai Dataset lain atau `git clone`.
2. Di notebook:
```bash
pip install -q -r requirements.txt
export EMO_DATA=/kaggle/input/<nama-dataset>   # folder berisi CSV
export EMO_OUT=/kaggle/working
cd <folder-kode>
python 01_audit.py
python 02_train_eval.py                         # ±30–60 menit di GPU (5 lipatan BERT)
python 03_label_corpus.py --sample 200000       # uji dulu; hapus --sample untuk seluruh korpus
python 04_topics.py --per-emotion 20000
python 05_diagnostics.py                        # diagnostik validitas (cepat, CPU)
```
3. Unduh folder `results/` dan `models/`, taruh di sebelah `app.py`, lalu: `streamlit run app.py`.

Tips:
- `03` aman diputus: hasil per 250 ribu tweet disimpan di `results/chunks/` dan dilanjutkan otomatis saat dijalankan ulang dengan argumen yang sama.
- Seluruh korpus 3,1 juta dengan IndoBERTweet memakan waktu lama (perkiraan 1–2 jam di T4 dengan fp16; ukur dulu dengan `--sample`). Jika tidak cukup, laporkan ukuran sampel (sudah ditampilkan dashboard).
- Tanpa GPU: `python 02_train_eval.py --skip-bert` tetap jalan (hanya model klasik), tetapi H1 tidak dapat diuji.
- Untuk membagikan dashboard lewat link: deploy ke Streamlit Community Cloud (`results/` yang kecil saja; `corpus_labeled.parquet` tidak perlu diunggah, halaman yang dipakai hanya butuh JSON/CSV kecil).

## Hal yang perlu Anda cek saat pertama dijalankan
- **Format file** sudah diverifikasi pada file asli: ketiga CSV berkolom `(indeks tanpa nama), tweet[, label]`; Lexicon berisi satu kolom label berisi nama emosi dan baris yang sama dengan Human.
- **Pembersihan Human**: 2.243 baris → 2.200 tweet unik (25 teks muncul ganda; 14 di antaranya berlabel bertentangan sehingga dibuang). Angka ini menggantikan "2.243" pada analisis.
- **Pencocokan Human–Lexicon** dilakukan lewat teks tweet yang dinormalisasi, bukan kolom indeks. Jumlah yang cocok tampil di halaman Audit; harusnya mendekati 2.243.
- **Hipotesis** diputuskan otomatis oleh `app.py` dari angka (H1: macro-F1 lebih tinggi **dan** McNemar p < 0,05; H2: CI95% selisih κ tidak mencakup 0; H3: p < 0,05, dengan Cramér's V ditampilkan). Jika hasil tidak mendukung hipotesis, dashboard menyatakan demikian; jangan diubah.
- Model klasik Naive Bayes tidak mendukung pembobotan kelas; pembobotan hanya berlaku pada SVM, LogReg, dan IndoBERTweet. Sebutkan ini di laporan.
- Skor keyakinan SVM adalah softmax dari margin, bukan probabilitas terkalibrasi.
- Hiperparameter BERT (lr 3e-5, batch 16, 4 epoch, panjang 96) berada di rentang proposal; catat nilai final di Bab hasil.
- Ambang keyakinan dihitung dari prediksi out-of-fold (pengganti data validasi) dengan presisi target 0,90 (`--target-precision`).

## Referensi metode tambahan (untuk daftar pustaka bila dipakai)
Kata khas per emosi memakai log-odds berbobot dengan prior Dirichlet: Monroe, B. L., Colaresi, M. P., & Quinn, K. M. (2008). Fightin' words. *Political Analysis, 16*(4), 372–403.
Cramér's V terkoreksi bias: Bergsma, W. (2013). A bias-correction for Cramér's V and Tschuprow's T. *Journal of the Korean Statistical Society, 42*(3), 323–328.

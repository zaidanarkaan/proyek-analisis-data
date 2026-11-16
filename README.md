# Proyek Analisis Data: E-Commerce Public Dataset

Submission proyek akhir **Belajar Analisis Data dengan Python** menggunakan **E-Commerce Public Dataset**.

## Struktur Submission
```text
submission/
├── dashboard/
│   ├── main_data.csv             # Dataset bersih hasil notebook
│   └── dashboard.py              # Aplikasi, fungsi RFM, dan peta Brasil
├── data/                        # Delapan CSV asli E-Commerce yang dibaca notebook
│   ├── customers_dataset.csv
│   ├── orders_dataset.csv
│   ├── order_items_dataset.csv
│   ├── order_payments_dataset.csv
│   ├── order_reviews_dataset.csv
│   ├── products_dataset.csv
│   ├── product_category_name_translation.csv
│   └── geolocation_dataset.csv
├── notebook.ipynb
├── README.md
├── requirements.txt
└── url.txt                      # URL publik apabila dashboard di-deploy
```

CSV dalam `data/`: `customers_dataset.csv`, `orders_dataset.csv`, `order_items_dataset.csv`, `order_payments_dataset.csv`, `order_reviews_dataset.csv`, `products_dataset.csv`, `product_category_name_translation.csv`, dan `geolocation_dataset.csv`. Notebook membacanya melalui path relatif `data/...`.

Folder `dashboard/` berisi tepat dua berkas: `main_data.csv` dan `dashboard.py`. Fungsi RFM, fungsi peta, dan koordinat batas Brasil berada di dalam `dashboard.py`; notebook mengimpor fungsi analisis tersebut tanpa menjalankan antarmuka Streamlit.

## Setup Environment — Anaconda
Jalankan dari direktori utama proyek atau direktori `submission/` hasil ekstraksi ZIP:
```bash
conda create --name ecommerce-ds python=3.12
conda activate ecommerce-ds
python -m pip install -r requirements.txt
```

## Setup Environment — Shell/Terminal
Gunakan virtual environment Python 3.12:
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```
Pada Windows, aktivasi environment menggunakan `.venv\Scripts\activate`.

## Menjalankan Dashboard Secara Lokal
Setelah mengaktifkan salah satu environment di atas:
```bash
python -m streamlit run dashboard/dashboard.py
```
Buka alamat yang ditampilkan Streamlit, biasanya `http://localhost:8501`. Dataset dashboard sudah tersedia dalam `dashboard/main_data.csv`; filter tanggal, state, dan kategori produk berada di sidebar.

## Menjalankan Ulang Notebook
Daftarkan kernel environment, kemudian jalankan seluruh notebook dari direktori utama proyek:
```bash
python -m ipykernel install --user --name ecommerce-ds --display-name "Python (ecommerce-ds)"
python -m nbconvert --to notebook --execute --inplace notebook.ipynb --ExecutePreprocessor.timeout=180 --ExecutePreprocessor.kernel_name=ecommerce-ds
```
Di Jupyter, pilih kernel **Python (ecommerce-ds)** lalu **Run All**. Proses ini memperbarui output analisis dan `dashboard/main_data.csv`.

## Membuat Berkas untuk Diunggah dari Workspace Pengembangan
Script `build_submission.py` tersedia di direktori utama workspace pengembangan. Siapkan folder data yang digunakan oleh notebook:
```bash
python build_submission.py --prepare-data
```
Setelah seluruh cell notebook dijalankan tanpa error, siapkan folder dan arsip submission:
```bash
python build_submission.py
```
Perintah tersebut menghasilkan folder **`submission/`** dengan struktur di atas dan arsip **`submission_ecommerce.zip`**. Unggah arsip tersebut. Manifest mencakup delapan CSV asli E-Commerce, notebook beserta output, README, requirements, dua berkas dashboard, dan URL publik. Semua path data dapat digunakan dari folder `submission/` hasil ekstraksi.

## Definisi Metrik
- Revenue bulanan dan kategori menggunakan `price` (harga produk tanpa ongkir).
- Monetary RFM dan revenue geografis menggunakan `price + freight_value`, sesuai label grafik.
- `is_late` membandingkan **tanggal kalender** penerimaan dan estimasi. Penerimaan pada tanggal estimasi termasuk tepat waktu; `delay_days` bernilai nol pada hari yang sama.
- Skor ulasan dihitung sekali per pesanan unik. Populasi ulasan yang memiliki skor valid disiapkan pada tahap Cleaning; metrik penjualan tetap mencakup pesanan tanpa ulasan.
- Recency memakai hari kalender sejak pembelian terakhir. Skor R memakai batas 30/90/180 hari, skor F memakai jumlah transaksi, dan skor M memakai kuartil dengan penanganan nilai seri yang menjaga skor pengeluaran identik tetap sama.
- Peta memakai satu lokasi kode pos terakhir yang valid per pelanggan unik; legenda menunjukkan pelanggan per sel heksagonal dengan skala logaritmik.

## Dashboard Streamlit Cloud
URL publik: https://9jfczyywa4ncnvsqn2x4ci.streamlit.app

URL tersebut dicatat dalam `url.txt`. Untuk pembaruan deployment, unggah perubahan ke repositori yang terhubung dengan Streamlit Cloud dan gunakan **Main file path** `dashboard/dashboard.py`, beserta `requirements.txt` dan `dashboard/main_data.csv`.

## Sumber Batas Peta
Konstanta `BRAZIL_OUTLINE` di `dashboard/dashboard.py` menggunakan geometri Brasil dari [Natural Earth — Admin 0 Countries 1:110m](https://github.com/nvkelso/natural-earth-vector/blob/master/geojson/ne_110m_admin_0_countries.geojson), dibulatkan hingga enam desimal. Data batas peta berada dalam [domain publik](https://www.naturalearthdata.com/about/terms-of-use/).

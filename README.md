# Proyek Analisis Data: E-Commerce Public Dataset

Submission proyek akhir **Belajar Analisis Data dengan Python** menggunakan **E-Commerce Public Dataset**.

## Struktur Direktori
- `notebook.ipynb` — Notebook analisis data lengkap (*Data Wrangling*, *EDA*, *Visualization & Explanatory Analysis*, *RFM Analysis*, *Clustering / Manual Grouping*, *Geospatial Analysis*, dan *Conclusion & Recommendation*).
- `dashboard/dashboard.py` — Aplikasi *dashboard* interaktif berbasis **Streamlit**.
- `dashboard/main_data.csv` — Dataset bersih hasil proses *Data Wrangling* yang digunakan oleh *dashboard*.
- `dashboard/geo_utils.py` — Helper peta kepadatan pelanggan unik yang digunakan oleh *notebook* dan *dashboard*.
- `dashboard/brazil_boundary.geojson` — Batas Brasil dari **Natural Earth 1:110m**, disimpan lokal untuk memberi konteks peta.
- `requirements.txt` — Daftar dependensi Python untuk menjalankan *notebook* dan *dashboard*.
- `url.txt` — Alamat dashboard saat ini: `http://localhost:8501` (penggunaan lokal).

## Definisi Metrik
- Revenue bulanan dan kategori dihitung dari `price` (harga produk tanpa ongkir).
- Monetary RFM dan revenue per negara bagian mencakup `price + freight_value` (produk + ongkir), sesuai label grafik.
- Setiap pesanan unik memiliki bobot yang sama dalam rata-rata ulasan; pesanan tanpa ulasan dikeluarkan dari perhitungan rata-rata.
- Peta menghitung satu lokasi kode pos terakhir yang valid per pelanggan unik. Warna menunjukkan jumlah pelanggan per sel heksagonal dengan skala logaritmik.

## Cara Menjalankan Dashboard Secara Lokal

1. Pastikan dependensi telah terinstal:
   ```bash
   pip install -r requirements.txt
   ```
2. Jalankan aplikasi Streamlit:
   ```bash
   streamlit run dashboard/dashboard.py
   ```

## Menjalankan Ulang Notebook
Jalankan dari direktori utama proyek setelah menginstal `requirements.txt`:
```bash
python -m nbconvert --to notebook --execute --inplace notebook.ipynb --ExecutePreprocessor.timeout=180 --ExecutePreprocessor.kernel_name=python3
```
Perintah ini memperbarui seluruh output *notebook* dan menghasilkan ulang `dashboard/main_data.csv` dari **E-Commerce Public Dataset**.

## Deploy ke Streamlit Cloud
Saat ini dashboard digunakan secara **lokal**. Untuk *deploy*:
1. Unggah (*push*) repositori ini ke GitHub, termasuk `notebook.ipynb`, `requirements.txt`, dan seluruh isi direktori `dashboard/`.
2. Buka [share.streamlit.io](https://share.streamlit.io/), hubungkan repositori GitHub Anda, lalu pilih **Main file path**: `dashboard/dashboard.py`.
3. Klik **Deploy**, lalu ganti alamat lokal dalam `url.txt` dengan URL publik Streamlit Cloud untuk submission yang menyertakan dashboard ter-*deploy*.

## Sumber Batas Peta
`dashboard/brazil_boundary.geojson` menggunakan geometri Brasil dari [Natural Earth — Admin 0 Countries 1:110m](https://github.com/nvkelso/natural-earth-vector/blob/master/geojson/ne_110m_admin_0_countries.geojson), dibulatkan hingga enam desimal. Data batas peta berada dalam [domain publik](https://www.naturalearthdata.com/about/terms-of-use/).

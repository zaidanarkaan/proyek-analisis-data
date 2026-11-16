from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from geo_utils import draw_customer_density_map

# Konfigurasi halaman
st.set_page_config(
    page_title="E-Commerce Analytics Dashboard",
    page_icon="🛒",
    layout="wide",
)

sns.set_style("whitegrid")
plt.rcParams["axes.edgecolor"] = "#CCCCCC"
plt.rcParams["axes.linewidth"] = 0.8


@st.cache_data
def load_data() -> pd.DataFrame:
    """Memuat dan memproses dataset utama dashboard."""
    base_dir = Path(__file__).resolve().parent
    data_path = base_dir / "main_data.csv"
    df = pd.read_csv(data_path)
    df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"])
    df["order_delivered_customer_date"] = pd.to_datetime(df["order_delivered_customer_date"])
    df["order_estimated_delivery_date"] = pd.to_datetime(df["order_estimated_delivery_date"])
    df["is_late"] = df["is_late"].astype(bool)
    return df


def compute_rfm(df: pd.DataFrame) -> pd.DataFrame:
    """Menghitung metrik RFM dan segmentasi Manual Grouping/Binning tanpa ML."""
    if df.empty:
        return pd.DataFrame()

    snapshot_date = df["order_purchase_timestamp"].max() + pd.Timedelta(days=1)
    rfm = (
        df.groupby("customer_unique_id", as_index=False)
        .agg(
            last_purchase=("order_purchase_timestamp", "max"),
            frequency=("order_id", "nunique"),
            monetary=("total_item_value", "sum"),
        )
    )
    rfm["recency"] = (snapshot_date - rfm["last_purchase"]).dt.days

    # Batas kuantil berbasis nilai sama dengan notebook; ranking hanya sebagai
    # fallback saat subset filter memiliki terlalu banyak nilai yang sama.
    if len(rfm) >= 4:
        for metric, score, labels in [
            ("recency", "r_score", [4, 3, 2, 1]),
            ("monetary", "m_score", [1, 2, 3, 4]),
        ]:
            try:
                rfm[score] = pd.qcut(rfm[metric], q=4, labels=labels).astype(int)
            except ValueError:
                rfm[score] = pd.qcut(rfm[metric].rank(method="first"), q=4, labels=labels).astype(int)
    else:
        rfm["r_score"] = 2
        rfm["m_score"] = 2

    rfm["f_score"] = np.where(rfm["frequency"] >= 3, 4, np.where(rfm["frequency"] == 2, 3, 1))
    rfm["rfm_score"] = rfm["r_score"] + rfm["f_score"] + rfm["m_score"]

    def assign_segment(row):
        if row["frequency"] >= 2 and row["r_score"] >= 3 and row["m_score"] >= 3:
            return "Champions"
        elif row["frequency"] >= 2:
            return "Loyal Repeaters"
        elif row["r_score"] >= 3 and row["m_score"] >= 3:
            return "Potential High-Value"
        elif row["r_score"] <= 2 and row["m_score"] >= 3:
            return "At Risk High-Value"
        elif row["r_score"] >= 3:
            return "Recent Customers"
        else:
            return "Hibernating"

    rfm["customer_segment"] = rfm.apply(assign_segment, axis=1)
    return rfm


all_df = load_data()

# ==================== SIDEBAR ====================
with st.sidebar:
    st.title("🛒 Filter Analisis")
    st.caption("Sesuaikan rentang waktu, wilayah, dan kategori produk.")

    min_date = all_df["order_purchase_timestamp"].min().date()
    max_date = all_df["order_purchase_timestamp"].max().date()

    date_range = st.date_input(
        "Rentang Tanggal Pembelian",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date, end_date = min_date, max_date

    states = sorted(all_df["customer_state"].dropna().unique().tolist())
    selected_states = st.multiselect(
        "Negara Bagian (State)",
        options=states,
        default=[],
        placeholder="Semua Negara Bagian",
    )

    categories = sorted(all_df["product_category_name_english"].dropna().unique().tolist())
    selected_categories = st.multiselect(
        "Kategori Produk",
        options=categories,
        default=[],
        placeholder="Semua Kategori Produk",
    )

# Filter dataframe
filtered_df = all_df[
    (all_df["order_purchase_timestamp"].dt.date >= start_date) &
    (all_df["order_purchase_timestamp"].dt.date <= end_date)
].copy()

if selected_states:
    filtered_df = filtered_df[filtered_df["customer_state"].isin(selected_states)]

if selected_categories:
    filtered_df = filtered_df[filtered_df["product_category_name_english"].isin(selected_categories)]

# ==================== HEADER & KPI ====================
st.title("📊 E-Commerce Public Dataset Analytics Dashboard")
st.markdown(
    "Dashboard interaktif untuk memantau **tren penjualan**, **kepuasan pelanggan berdasarkan ketepatan pengiriman**, "
    "**segmentasi pelanggan (RFM)**, dan **sebaran geografis**."
)

if filtered_df.empty:
    st.warning("Tidak ada data yang sesuai dengan kombinasi filter yang dipilih. Silakan ubah filter di sidebar.")
    st.stop()

order_summary_df = filtered_df.groupby("order_id", as_index=False).agg(
    customer_unique_id=("customer_unique_id", "first"),
    customer_state=("customer_state", "first"),
    is_late=("is_late", "first"),
    review_score=("review_score", "first"),
    total_item_value=("total_item_value", "sum"),
)

total_orders = filtered_df["order_id"].nunique()
total_revenue = filtered_df["price"].sum()
avg_review = order_summary_df["review_score"].mean()
on_time_rate = (1 - order_summary_df["is_late"].mean()) * 100

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Pesanan Unik", f"{total_orders:,}")
col2.metric("Total Revenue (Produk)", f"R$ {total_revenue:,.0f}")
col3.metric("Rata-rata Skor Ulasan", f"{avg_review:.2f} / 5.00" if pd.notna(avg_review) else "Tidak tersedia")
col4.metric("Ketepatan Pengiriman", f"{on_time_rate:.1f}%")
st.caption("Revenue penjualan = harga produk (price), tanpa ongkir. Skor ulasan dihitung sekali per pesanan unik.")

st.divider()

# ==================== TABS ====================
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Penjualan & Kategori",
    "🚚 Pengiriman & Ulasan",
    "👥 RFM & Segmentasi",
    "🗺️ Analisis Geografis",
])

# --- TAB 1: PENJUALAN & KATEGORI ---
with tab1:
    st.subheader("Tren Bulanan & Kategori dengan Revenue Tertinggi dan Terendah")

    filtered_df["year_month"] = filtered_df["order_purchase_timestamp"].dt.to_period("M").astype(str)
    monthly_df = (
        filtered_df.groupby("year_month", as_index=False)
        .agg(total_orders=("order_id", "nunique"), total_revenue=("price", "sum"))
        .sort_values("year_month")
    )

    cat_df = (
        filtered_df.groupby("product_category_name_english", as_index=False)
        .agg(total_orders=("order_id", "nunique"), total_revenue=("price", "sum"))
        .sort_values("total_revenue", ascending=False)
    )

    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        x = np.arange(len(monthly_df))
        peak_pos = int(monthly_df["total_orders"].to_numpy().argmax())
        bars = ax.bar(x, monthly_df["total_orders"], color="#B0C4DE", width=0.65)
        bars[peak_pos].set_color("#1F4E78")
        peak_orders = monthly_df.iloc[peak_pos]["total_orders"]
        ax.annotate(f"Puncak: {peak_orders:,}\n{monthly_df.iloc[peak_pos]['year_month']}",
                    xy=(peak_pos, peak_orders), xytext=(0, 8), textcoords="offset points",
                    ha="center", fontsize=9, fontweight="bold", color="#1F4E78")
        ax.set_xticks(x)
        ax.set_xticklabels(monthly_df["year_month"], rotation=45, ha="right", fontsize=8)
        ax.set_ylim(0, monthly_df["total_orders"].max() * 1.24)
        ax.set_ylabel("Jumlah Pesanan Unik", fontweight="bold")
        ax.set_title("Tren Jumlah Pesanan Bulanan", fontweight="bold")
        ax.grid(axis="x", visible=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with c2:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        revenue_k = monthly_df["total_revenue"] / 1e3
        peak_revenue_pos = int(revenue_k.to_numpy().argmax())
        ax.plot(x, revenue_k, color="#D95F02", marker="o", linewidth=2)
        ax.annotate(f"Puncak: R$ {revenue_k.iloc[peak_revenue_pos]:,.0f}k\n{monthly_df.iloc[peak_revenue_pos]['year_month']}",
                    xy=(peak_revenue_pos, revenue_k.iloc[peak_revenue_pos]),
                    xytext=(0, 8), textcoords="offset points", ha="center", fontsize=9,
                    fontweight="bold", color="#D95F02")
        ax.set_xticks(x)
        ax.set_xticklabels(monthly_df["year_month"], rotation=45, ha="right", fontsize=8)
        ax.set_ylim(0, revenue_k.max() * 1.24)
        ax.set_ylabel("Revenue Produk (Ribu BRL)", fontweight="bold")
        ax.set_title("Tren Revenue Produk Bulanan", fontweight="bold")
        ax.grid(axis="x", visible=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    c1, c2 = st.columns(2)
    with c1:
        top10_cat = cat_df.head(10).sort_values("total_revenue", ascending=True)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        colors_cat = ["#1F4E78" if value == cat_df["total_revenue"].max() else "#B0C4DE"
                      for value in top10_cat["total_revenue"]]
        ax.barh(top10_cat["product_category_name_english"], top10_cat["total_revenue"] / 1e3, color=colors_cat)
        ax.set_xlabel("Total Revenue (Ribu BRL)", fontweight="bold")
        ax.set_title(f"Top {len(top10_cat)} Kategori dengan Revenue Tertinggi", fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlim(0, top10_cat["total_revenue"].max() / 1e3 * 1.22)
        ax.grid(axis="y", visible=False)
        for i, val in enumerate(top10_cat["total_revenue"] / 1e3):
            ax.text(val + (top10_cat["total_revenue"].max() / 1e3 * 0.02), i, f"R$ {val:,.0f}k", va="center", fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with c2:
        bottom5_cat = cat_df.tail(5).sort_values("total_revenue", ascending=False)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        bottom_colors = ["#C0392B" if value == cat_df["total_revenue"].min() else "#B0C4DE"
                         for value in bottom5_cat["total_revenue"]]
        ax.barh(bottom5_cat["product_category_name_english"], bottom5_cat["total_revenue"], color=bottom_colors)
        ax.set_xlabel("Total Revenue (BRL)", fontweight="bold")
        ax.set_title(f"Bottom {len(bottom5_cat)} Kategori dengan Revenue Terendah", fontweight="bold")
        ax.set_xlim(0, bottom5_cat["total_revenue"].max() * 1.35)
        ax.grid(axis="y", visible=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for i, value in enumerate(bottom5_cat["total_revenue"]):
            ax.text(value + bottom5_cat["total_revenue"].max() * 0.02, i,
                    f"R$ {value:,.2f}", va="center", fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    highest_category = cat_df.iloc[0]
    lowest_category = cat_df.iloc[-1]
    st.markdown(
        f"**Kategori tertinggi:** {highest_category['product_category_name_english']} "
        f"(R$ {highest_category['total_revenue']:,.2f}). "
        f"**Kategori terendah:** {lowest_category['product_category_name_english']} "
        f"(R$ {lowest_category['total_revenue']:,.2f})."
    )
    st.caption("Peringkat mencakup kategori yang memiliki pesanan pada filter aktif. Grafik Top memakai ribu BRL; Bottom memakai BRL.")

# --- TAB 2: PENGIRIMAN & ULASAN ---
with tab2:
    st.subheader("Pengaruh Ketepatan Pengiriman terhadap Skor Ulasan")
    st.caption("Setiap order_id memiliki bobot yang sama. Pesanan tanpa skor ulasan dikeluarkan dari perbandingan ulasan.")

    deliv_clean = order_summary_df.dropna(subset=["review_score"]).copy()
    if deliv_clean.empty:
        st.info("Tidak ada skor ulasan yang tersedia pada filter aktif. Ubah filter untuk melihat perbandingan.")
    else:
        deliv_clean["delivery_status"] = np.where(
            deliv_clean["is_late"], "Terlambat (Late)", "Tepat Waktu / Lebih Awal"
        )
        deliv_clean["is_negative"] = deliv_clean["review_score"] <= 2

        deliv_summary = (
            deliv_clean.groupby("delivery_status", as_index=False)
            .agg(
                total_orders=("order_id", "count"),
                avg_score=("review_score", "mean"),
                neg_pct=("is_negative", lambda s: s.mean() * 100),
            )
            .sort_values("avg_score", ascending=False)
        )

        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(7, 4.5))
            colors_d = ["#2E8B57" if "Tepat" in s else "#C0392B" for s in deliv_summary["delivery_status"]]
            bars = ax.bar(deliv_summary["delivery_status"], deliv_summary["avg_score"], color=colors_d, width=0.5)
            ax.set_ylim(0, 5.3)
            ax.set_ylabel("Rata-rata Review Score (1–5)", fontweight="bold")
            ax.set_title("Rata-rata Skor Ulasan: Tepat Waktu vs Terlambat", fontweight="bold")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            for bar, (_, row) in zip(bars, deliv_summary.iterrows()):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.12,
                    f"{row['avg_score']:.2f}\n({row['neg_pct']:.1f}% skor 1–2)",
                    ha="center", va="bottom", fontsize=9, fontweight="bold"
                )
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        with c2:
            dist_df = (
                deliv_clean.assign(score_int=deliv_clean["review_score"].round().astype(int))
                .groupby(["delivery_status", "score_int"])["order_id"]
                .count()
                .unstack(fill_value=0)
            )
            for s in range(1, 6):
                if s not in dist_df.columns:
                    dist_df[s] = 0
            dist_df = dist_df[[1, 2, 3, 4, 5]]
            dist_pct = dist_df.div(dist_df.sum(axis=1), axis=0) * 100

            fig, ax = plt.subplots(figsize=(7, 4.5))
            x_scores = np.arange(1, 6)
            w = 0.35
            if "Tepat Waktu / Lebih Awal" in dist_pct.index:
                ax.bar(x_scores - w/2, dist_pct.loc["Tepat Waktu / Lebih Awal"], width=w, color="#2E8B57", label="Tepat Waktu / Lebih Awal")
            if "Terlambat (Late)" in dist_pct.index:
                ax.bar(x_scores + w/2, dist_pct.loc["Terlambat (Late)"], width=w, color="#C0392B", label="Terlambat (Late)")
            ax.set_xticks(x_scores)
            ax.set_xticklabels([f"Skor {s}" for s in x_scores])
            ax.set_ylabel("Persentase Pesanan (%)", fontweight="bold")
            ax.set_title("Distribusi Skor Ulasan Berdasarkan Ketepatan Pengiriman", fontweight="bold")
            ax.legend(frameon=False)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        st.caption("Skor rata-rata per pesanan dibulatkan ke integer hanya untuk grafik distribusi; rata-rata dan proporsi negatif memakai skor aslinya.")

# --- TAB 3: RFM & SEGMENTASI ---
with tab3:
    st.subheader("RFM Analysis & Segmentasi Pelanggan (Manual Grouping)")

    rfm_df = compute_rfm(filtered_df)
    seg_summary = (
        rfm_df.groupby("customer_segment", as_index=False)
        .agg(
            customer_count=("customer_unique_id", "count"),
            avg_recency=("recency", "mean"),
            avg_frequency=("frequency", "mean"),
            avg_monetary=("monetary", "mean"),
        )
    )
    seg_summary["customer_pct"] = seg_summary["customer_count"] / seg_summary["customer_count"].sum() * 100

    m1, m2, m3 = st.columns(3)
    m1.metric("Rata-rata Recency (Hari)", f"{rfm_df['recency'].mean():.1f}")
    m2.metric("Rata-rata Frequency (Order)", f"{rfm_df['frequency'].mean():.2f}")
    m3.metric("Rata-rata Monetary (BRL)", f"R$ {rfm_df['monetary'].mean():,.2f}")

    c1, c2 = st.columns(2)
    with c1:
        seg_plot = seg_summary.sort_values("customer_count", ascending=True)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        colors_seg = ["#B0C4DE"] * max(0, len(seg_plot) - 1) + ["#1F4E78"]
        ax.barh(seg_plot["customer_segment"], seg_plot["customer_count"], color=colors_seg[:len(seg_plot)])
        ax.set_xlabel("Jumlah Pelanggan Unik", fontweight="bold")
        ax.set_title("Distribusi Segmen Pelanggan", fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlim(0, seg_plot["customer_count"].max() * 1.24)
        for i, (_, row) in enumerate(seg_plot.iterrows()):
            ax.text(row["customer_count"] + seg_plot["customer_count"].max() * 0.02, i,
                    f"{row['customer_count']:,} ({row['customer_pct']:.1f}%)", va="center", fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with c2:
        seg_mon = seg_summary.sort_values("avg_monetary", ascending=True)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        mon_colors = ["#1F4E78" if value == seg_mon["avg_monetary"].max() else "#B0C4DE"
                      for value in seg_mon["avg_monetary"]]
        ax.barh(seg_mon["customer_segment"], seg_mon["avg_monetary"], color=mon_colors)
        ax.set_xlabel("Rata-rata Monetary (BRL)", fontweight="bold")
        ax.set_title("Rata-rata Nilai Belanja per Segmen", fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlim(0, seg_mon["avg_monetary"].max() * 1.2)
        for i, val in enumerate(seg_mon["avg_monetary"]):
            ax.text(val + seg_mon["avg_monetary"].max() * 0.02, i, f"R$ {val:,.1f}", va="center", fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.caption("Biru tua menandai nilai tertinggi pada masing-masing grafik. Monetary = harga produk + ongkir; skor RFM dihitung ulang pada filter aktif.")

# --- TAB 4: GEOSPATIAL ---
with tab4:
    st.subheader("Geospatial Analysis: Sebaran Pelanggan & Revenue per Negara Bagian")
    st.caption("Revenue wilayah mencakup harga produk + ongkir. Rata-rata ulasan dihitung per pesanan unik, bukan per item.")

    state_summary = (
        order_summary_df.groupby("customer_state", as_index=False)
        .agg(
            total_customers=("customer_unique_id", "nunique"),
            total_orders=("order_id", "count"),
            total_revenue=("total_item_value", "sum"),
            reviewed_orders=("review_score", "count"),
            avg_review_score=("review_score", "mean"),
        )
        .sort_values("total_revenue", ascending=False)
    )
    state_summary["share_pct"] = state_summary["total_revenue"] / state_summary["total_revenue"].sum() * 100

    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(7.5, 5))
        mapped_customers = draw_customer_density_map(ax, filtered_df)
        ax.set_title("Kepadatan Pelanggan Unik di Brasil", fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.caption(
            f"Terpetakan {mapped_customers:,} dari {filtered_df['customer_unique_id'].nunique():,} pelanggan unik. "
            "Lokasi memakai median koordinat kode pos dari pesanan terakhir yang memiliki koordinat valid. "
            "Warna menunjukkan pelanggan per sel (skala log); batas Brasil: Natural Earth 1:110m."
        )

    with c2:
        top10_states = state_summary.head(10).sort_values("total_revenue", ascending=True)
        fig, ax = plt.subplots(figsize=(7.5, 5))
        st_colors = ["#B0C4DE"] * max(0, len(top10_states) - 1) + ["#1F4E78"]
        ax.barh(top10_states["customer_state"], top10_states["total_revenue"] / 1e6, color=st_colors[:len(top10_states)])
        ax.set_xlabel("Revenue Produk + Ongkir (Juta BRL)", fontweight="bold")
        ax.set_title("Top 10 Negara Bagian berdasarkan Revenue termasuk Ongkir", fontweight="bold", fontsize=11)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if not top10_states.empty:
            ax.set_xlim(0, top10_states["total_revenue"].max() / 1e6 * 1.24)
            for i, (_, row) in enumerate(top10_states.iterrows()):
                ax.text(
                    row["total_revenue"] / 1e6 + (top10_states["total_revenue"].max() / 1e6 * 0.02),
                    i,
                    f"R$ {row['total_revenue']/1e6:.2f}M ({row['share_pct']:.1f}%)",
                    va="center", fontsize=8
                )
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.dataframe(
        state_summary.rename(columns={
            "customer_state": "State",
            "total_customers": "Pelanggan Unik",
            "total_orders": "Pesanan Unik",
            "total_revenue": "Revenue Produk + Ongkir (BRL)",
            "reviewed_orders": "Pesanan dengan Ulasan",
            "avg_review_score": "Rata-rata Skor Ulasan",
            "share_pct": "Kontribusi Revenue (%)",
        }).style.format({
            "Revenue Produk + Ongkir (BRL)": "{:,.2f}",
            "Rata-rata Skor Ulasan": "{:.2f}",
            "Kontribusi Revenue (%)": "{:.1f}",
        }, na_rep="Tidak tersedia"),
        hide_index=True,
    )

st.caption("© Submission Proyek Analisis Data dengan Python — E-Commerce Public Dataset")

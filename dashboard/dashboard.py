"""Dashboard E-Commerce mandiri dengan fungsi analisis yang dapat diimpor notebook."""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm
from matplotlib.patches import Polygon
from matplotlib.ticker import FuncFormatter


# Batas Brasil: Natural Earth Admin 0 Countries 1:110m, domain publik.
# Sumber: https://github.com/nvkelso/natural-earth-vector/blob/master/geojson/ne_110m_admin_0_countries.geojson
# Koordinat dibulatkan hingga enam desimal dan disimpan di sini agar dashboard
# hanya membutuhkan dashboard.py dan main_data.csv.
BRAZIL_OUTLINE = np.array([
    [-57.625133, -30.216295], [-56.2909, -28.852761],
    [-55.162286, -27.881915], [-54.490725, -27.474757],
    [-53.648735, -26.923473], [-53.628349, -26.124865],
    [-54.13005, -25.547639], [-54.625291, -25.739255],
    [-54.428946, -25.162185], [-54.293476, -24.5708],
    [-54.29296, -24.021014], [-54.652834, -23.839578],
    [-55.027902, -24.001274], [-55.400747, -23.956935],
    [-55.517639, -23.571998], [-55.610683, -22.655619],
    [-55.797958, -22.35693], [-56.473317, -22.0863],
    [-56.88151, -22.282154], [-57.937156, -22.090176],
    [-57.870674, -20.732688], [-58.166392, -20.176701],
    [-57.853802, -19.969995], [-57.949997, -19.400004],
    [-57.676009, -18.96184], [-57.498371, -18.174188],
    [-57.734558, -17.552468], [-58.280804, -17.27171],
    [-58.388058, -16.877109], [-58.24122, -16.299573],
    [-60.15839, -16.258284], [-60.542966, -15.09391],
    [-60.251149, -15.077219], [-60.264326, -14.645979],
    [-60.459198, -14.354007], [-60.503304, -13.775955],
    [-61.084121, -13.479384], [-61.713204, -13.489202],
    [-62.127081, -13.198781], [-62.80306, -13.000653],
    [-63.196499, -12.627033], [-64.316353, -12.461978],
    [-65.402281, -11.56627], [-65.321899, -10.895872],
    [-65.444837, -10.511451], [-65.338435, -9.761988],
    [-66.646908, -9.931331], [-67.173801, -10.306812],
    [-68.048192, -10.712059], [-68.271254, -11.014521],
    [-68.786158, -11.03638], [-69.529678, -10.951734],
    [-70.093752, -11.123972], [-70.548686, -11.009147],
    [-70.481894, -9.490118], [-71.302412, -10.079436],
    [-72.184891, -10.053598], [-72.563033, -9.520194],
    [-73.226713, -9.462213], [-73.015383, -9.032833],
    [-73.571059, -8.424447], [-73.987235, -7.52383],
    [-73.723401, -7.340999], [-73.724487, -6.918595],
    [-73.120027, -6.629931], [-73.219711, -6.089189],
    [-72.964507, -5.741251], [-72.891928, -5.274561],
    [-71.748406, -4.593983], [-70.928843, -4.401591],
    [-70.794769, -4.251265], [-69.893635, -4.298187],
    [-69.444102, -1.556287], [-69.420486, -1.122619],
    [-69.577065, -0.549992], [-70.020656, -0.185156],
    [-70.015566, 0.541414], [-69.452396, 0.706159],
    [-69.252434, 0.602651], [-69.218638, 0.985677],
    [-69.804597, 1.089081], [-69.816973, 1.714805],
    [-67.868565, 1.692455], [-67.53781, 2.037163],
    [-67.259998, 1.719999], [-67.065048, 1.130112],
    [-66.876326, 1.253361], [-66.325765, 0.724452],
    [-65.548267, 0.789254], [-65.354713, 1.095282],
    [-64.611012, 1.328731], [-64.199306, 1.492855],
    [-64.083085, 1.916369], [-63.368788, 2.2009],
    [-63.422867, 2.411068], [-64.269999, 2.497006],
    [-64.408828, 3.126786], [-64.368494, 3.79721],
    [-64.816064, 4.056445], [-64.628659, 4.148481],
    [-63.888343, 4.02053], [-63.093198, 3.770571],
    [-62.804533, 4.006965], [-62.08543, 4.162124],
    [-60.966893, 4.536468], [-60.601179, 4.918098],
    [-60.733574, 5.200277], [-60.213683, 5.244486],
    [-59.980959, 5.014061], [-60.111002, 4.574967],
    [-59.767406, 4.423503], [-59.53804, 3.958803],
    [-59.815413, 3.606499], [-59.974525, 2.755233],
    [-59.718546, 2.24963], [-59.646044, 1.786894],
    [-59.030862, 1.317698], [-58.540013, 1.268088],
    [-58.429477, 1.463942], [-58.11345, 1.507195],
    [-57.660971, 1.682585], [-57.335823, 1.948538],
    [-56.782704, 1.863711], [-56.539386, 1.899523],
    [-55.995698, 1.817667], [-55.9056, 2.021996],
    [-56.073342, 2.220795], [-55.973322, 2.510364],
    [-55.569755, 2.421506], [-55.097587, 2.523748],
    [-54.524754, 2.311849], [-54.088063, 2.105557],
    [-53.778521, 2.376703], [-53.554839, 2.334897],
    [-53.418465, 2.053389], [-52.939657, 2.124858],
    [-52.556425, 2.504705], [-52.249338, 3.241094],
    [-51.657797, 4.156232], [-51.317146, 4.203491],
    [-51.069771, 3.650398], [-50.508875, 1.901564],
    [-49.974076, 1.736483], [-49.947101, 1.04619],
    [-50.699251, 0.222984], [-50.388211, -0.078445],
    [-48.620567, -0.235489], [-48.584497, -1.237805],
    [-47.824956, -0.581618], [-46.566584, -0.941028],
    [-44.905703, -1.55174], [-44.417619, -2.13775],
    [-44.581589, -2.691308], [-43.418791, -2.38311],
    [-41.472657, -2.912018], [-39.978665, -2.873054],
    [-38.500383, -3.700652], [-37.223252, -4.820946],
    [-36.452937, -5.109404], [-35.597796, -5.149504],
    [-35.235389, -5.464937], [-34.89603, -6.738193],
    [-34.729993, -7.343221], [-35.128212, -8.996401],
    [-35.636967, -9.649282], [-37.046519, -11.040721],
    [-37.683612, -12.171195], [-38.423877, -13.038119],
    [-38.673887, -13.057652], [-38.953276, -13.79337],
    [-38.882298, -15.667054], [-39.161092, -17.208407],
    [-39.267339, -17.867746], [-39.583521, -18.262296],
    [-39.760823, -19.599113], [-40.774741, -20.904512],
    [-40.944756, -21.937317], [-41.754164, -22.370676],
    [-41.988284, -22.97007], [-43.074704, -22.967693],
    [-44.647812, -23.351959], [-45.352136, -23.796842],
    [-46.472093, -24.088969], [-47.648972, -24.885199],
    [-48.495458, -25.877025], [-48.641005, -26.623698],
    [-48.474736, -27.175912], [-48.66152, -28.186135],
    [-48.888457, -28.674115], [-49.587329, -29.224469],
    [-50.696874, -30.984465], [-51.576226, -31.777698],
    [-52.256081, -32.24537], [-52.7121, -33.196578],
    [-53.373662, -33.768378], [-53.650544, -33.202004],
    [-53.209589, -32.727666], [-53.787952, -32.047243],
    [-54.572452, -31.494511], [-55.60151, -30.853879],
    [-55.973245, -30.883076], [-56.976026, -30.109686],
    [-57.625133, -30.216295],
])


def score_recency(recency: pd.Series) -> pd.Series:
    """Skor operasional: ≤30, 31–90, 91–180, dan >180 hari kalender."""
    return pd.cut(
        recency, bins=[-1, 30, 90, 180, np.inf], labels=[4, 3, 2, 1]
    ).astype(int)


def score_monetary(monetary: pd.Series) -> pd.Series:
    """Kuartil belanja; nilai uang yang sama selalu memperoleh skor yang sama.

    Kohort <4 pelanggan atau belanja seragam memakai skor netral 2. Jika batas
    kuartil berulang, posisi persentil rank rata-rata digunakan untuk binning.
    """
    money = monetary.round(2)
    if len(money) < 4 or money.nunique() < 2:
        return pd.Series(2, index=money.index, dtype=int)
    try:
        return pd.qcut(money, q=4, labels=[1, 2, 3, 4]).astype(int)
    except ValueError:
        percentile = money.rank(method="average", pct=True)
        return pd.cut(
            percentile, bins=[0, 0.25, 0.5, 0.75, 1],
            labels=[1, 2, 3, 4], include_lowest=True,
        ).astype(int)


def draw_customer_density_map(ax, df) -> int:
    """Gambar satu lokasi kode pos terakhir yang valid per pelanggan unik.

    Warna menunjukkan jumlah pelanggan per sel heksagonal, bukan jumlah item.
    Nilai balik adalah jumlah pelanggan yang memiliki koordinat untuk dipetakan.
    """
    outline = BRAZIL_OUTLINE
    ax.set_facecolor("#EAF2F8")
    ax.add_patch(Polygon(outline, facecolor="#F4F6F8", edgecolor="none", zorder=1))

    geo_customers = (
        df.dropna(subset=["customer_unique_id", "geolocation_lat", "geolocation_lng"])
        .sort_values("order_purchase_timestamp")
        .drop_duplicates(subset=["customer_unique_id"], keep="last")
    )
    if geo_customers.empty:
        ax.text(0.5, 0.5, "Tidak ada koordinat pelanggan untuk filter ini.",
                transform=ax.transAxes, ha="center", va="center")
    else:
        density = ax.hexbin(
            geo_customers["geolocation_lng"],
            geo_customers["geolocation_lat"],
            gridsize=40,
            extent=(-75, -34, -35, 6),
            mincnt=1,
            cmap="YlOrRd",
            norm=LogNorm(vmin=1),
            linewidths=0,
            zorder=2,
        )
        max_count = max(2, int(density.get_array().max()))
        density.set_clim(1, max_count)
        colorbar = ax.figure.colorbar(density, ax=ax, fraction=0.046, pad=0.04)
        ticks = sorted({1, max_count, *[10 ** n for n in range(1, int(np.log10(max_count)) + 1)]})
        colorbar.set_ticks(ticks)
        colorbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
        colorbar.ax.minorticks_off()
        colorbar.set_label("Pelanggan unik per sel (skala log)", fontsize=9)

    ax.plot(outline[:, 0], outline[:, 1], color="#667085", linewidth=0.9, zorder=3)
    ax.text(-57, -7, "BRASIL", color="#667085", fontsize=10, fontweight="bold", zorder=4)
    ax.text(-38, -27, "Samudra\nAtlantik", color="#667085", fontsize=9,
            ha="center", style="italic", zorder=4)
    ax.set_xlim(-75, -33)
    ax.set_ylim(-35, 7)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Longitude", fontweight="bold")
    ax.set_ylabel("Latitude", fontweight="bold")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{abs(value):.0f}°W"))
    ax.yaxis.set_major_formatter(FuncFormatter(
        lambda value, _: f"{abs(value):.0f}°{'S' if value < 0 else 'N'}" if value else "0°"
    ))
    ax.grid(color="#D0D5DD", linewidth=0.5, alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return len(geo_customers)


def load_data() -> pd.DataFrame:
    """Memuat dan memproses dataset utama dashboard."""
    base_dir = Path(__file__).resolve().parent
    data_path = base_dir / "main_data.csv"
    df = pd.read_csv(data_path)
    df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"])
    df["order_delivered_customer_date"] = pd.to_datetime(df["order_delivered_customer_date"])
    df["order_estimated_delivery_date"] = pd.to_datetime(df["order_estimated_delivery_date"])
    purchase = df["order_purchase_timestamp"].dt.normalize()
    received = df["order_delivered_customer_date"].dt.normalize()
    estimated = df["order_estimated_delivery_date"].dt.normalize()
    df["delivery_days"] = (received - purchase).dt.days
    df["estimated_delivery_days"] = (estimated - purchase).dt.days
    df["delay_days"] = (received - estimated).dt.days
    df["is_late"] = df["delay_days"].gt(0)
    df["delivery_status"] = np.where(df["is_late"], "Terlambat (Late)", "Tepat Waktu / Lebih Awal")
    return df


def compute_rfm(df: pd.DataFrame) -> pd.DataFrame:
    """Menghitung metrik RFM dan segmentasi Manual Grouping/Binning tanpa ML."""
    if df.empty:
        return pd.DataFrame()

    snapshot_date = df["order_purchase_timestamp"].max().normalize() + pd.DateOffset(days=1)
    rfm = (
        df.groupby("customer_unique_id", as_index=False)
        .agg(
            last_purchase=("order_purchase_timestamp", "max"),
            frequency=("order_id", "nunique"),
            monetary=("total_item_value", "sum"),
        )
    )
    rfm["monetary"] = rfm["monetary"].round(2)
    rfm["recency"] = (snapshot_date - rfm["last_purchase"].dt.normalize()).dt.days
    rfm["r_score"] = score_recency(rfm["recency"])
    rfm["m_score"] = score_monetary(rfm["monetary"])

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


def main():
    """Jalankan antarmuka Streamlit saat dashboard.py menjadi entry point."""
    import streamlit as st

    st.set_page_config(
        page_title="E-Commerce Analytics Dashboard",
        page_icon="🛒",
        layout="wide",
    )
    sns.set_style("whitegrid")
    plt.rcParams["axes.edgecolor"] = "#CCCCCC"
    plt.rcParams["axes.linewidth"] = 0.8

    all_df = st.cache_data(load_data)()

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
        order_purchase_timestamp=("order_purchase_timestamp", "first"),
        customer_unique_id=("customer_unique_id", "first"),
        customer_state=("customer_state", "first"),
        is_late=("is_late", "first"),
        delivery_status=("delivery_status", "first"),
        delivery_days=("delivery_days", "first"),
        estimated_delivery_days=("estimated_delivery_days", "first"),
        delay_days=("delay_days", "first"),
        review_score=("review_score", "first"),
        total_item_value=("total_item_value", "sum"),
    )
    review_ready_order_df = order_summary_df.dropna(subset=["review_score"]).copy()
    review_ready_order_df["is_negative_review"] = review_ready_order_df["review_score"].le(2)
    review_ready_order_df["score_int"] = review_ready_order_df["review_score"].round().astype(int)

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
    st.caption(
        f"Periode pembelian aktif: {start_date:%d-%m-%Y} – {end_date:%d-%m-%Y}. "
        "Pengiriman pada tanggal estimasi termasuk tepat waktu; durasi menggunakan hari kalender."
    )

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
        month_index = pd.period_range(start=start_date, end=end_date, freq="M").astype(str)
        monthly_df = monthly_df.set_index("year_month").reindex(month_index, fill_value=0).reset_index(names="year_month")
        for value, growth in [("total_orders", "mom_order_growth_pct"), ("total_revenue", "mom_revenue_growth_pct")]:
            previous = monthly_df[value].shift().replace(0, np.nan)
            monthly_df[growth] = monthly_df[value].diff().div(previous) * 100

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
            ax.set_xlabel("Bulan Pembelian", fontweight="bold")
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
            ax.annotate(f"Puncak: R$ {revenue_k.iloc[peak_revenue_pos]:,.1f}k\n{monthly_df.iloc[peak_revenue_pos]['year_month']}",
                        xy=(peak_revenue_pos, revenue_k.iloc[peak_revenue_pos]),
                        xytext=(0, 8), textcoords="offset points", ha="center", fontsize=9,
                        fontweight="bold", color="#D95F02")
            ax.set_xticks(x)
            ax.set_xticklabels(monthly_df["year_month"], rotation=45, ha="right", fontsize=8)
            ax.set_ylim(0, revenue_k.max() * 1.24)
            ax.set_xlabel("Bulan Pembelian", fontweight="bold")
            ax.set_ylabel("Revenue Produk (Ribu BRL)", fontweight="bold")
            ax.set_title("Tren Revenue Produk Bulanan", fontweight="bold")
            ax.grid(axis="x", visible=False)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        if monthly_df[["mom_order_growth_pct", "mom_revenue_growth_pct"]].notna().any().any():
            fig, ax = plt.subplots(figsize=(13, 4))
            ax.plot(x, monthly_df["mom_order_growth_pct"], marker="o", color="#1F4E78", label="MoM Pesanan")
            ax.plot(x, monthly_df["mom_revenue_growth_pct"], marker="s", color="#D95F02", label="MoM Revenue")
            ax.axhline(0, color="#667085", linewidth=1)
            ax.set_xticks(x)
            ax.set_xticklabels(monthly_df["year_month"], rotation=45, ha="right", fontsize=8)
            ax.set_ylabel("Pertumbuhan terhadap Bulan Sebelumnya (%)", fontweight="bold")
            ax.set_xlabel("Bulan Pembelian", fontweight="bold")
            ax.set_title("Pertumbuhan Month-over-Month pada Filter Aktif", fontweight="bold")
            ax.legend(frameon=False)
            ax.grid(axis="x", visible=False)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        st.caption("MoM tidak tersedia pada bulan pertama atau ketika nilai bulan sebelumnya nol. Bulan tanpa pesanan ditampilkan sebagai nol; bulan batas mengikuti tanggal filter aktif.")

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
        st.subheader("Perbandingan Ulasan menurut Ketepatan Pengiriman")
        st.caption("Setiap order_id memiliki bobot yang sama. Pesanan tanpa skor ulasan dikeluarkan dari perbandingan ulasan.")

        deliv_clean = review_ready_order_df
        if deliv_clean.empty:
            st.info("Tidak ada skor ulasan yang tersedia pada filter aktif. Ubah filter untuk melihat perbandingan.")
        else:
            deliv_summary = (
                deliv_clean.groupby("delivery_status", as_index=False)
                .agg(
                    total_orders=("order_id", "count"),
                    avg_score=("review_score", "mean"),
                    neg_pct=("is_negative_review", lambda s: s.mean() * 100),
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
                    deliv_clean.groupby(["delivery_status", "score_int"])["order_id"]
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
                ax.set_title("Distribusi Skor Ulasan (Rata-rata per Order Dibulatkan)", fontweight="bold", fontsize=11)
                ax.legend(frameon=False)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

            st.caption("Skor rata-rata per pesanan dibulatkan ke integer hanya untuk grafik distribusi; rata-rata dan proporsi negatif memakai skor aslinya.")

        st.subheader("Diagnostik Pengiriman per State")
        delivery_state_df = (
            order_summary_df.groupby("customer_state", as_index=False)
            .agg(
                total_orders=("order_id", "count"),
                late_orders=("is_late", "sum"),
                late_rate_pct=("is_late", lambda s: s.mean() * 100),
                avg_delivery_days=("delivery_days", "mean"),
                avg_estimated_days=("estimated_delivery_days", "mean"),
                avg_delay_days=("delay_days", "mean"),
            )
            .sort_values(["late_rate_pct", "total_orders"], ascending=False)
        )
        priority_states = delivery_state_df[delivery_state_df["total_orders"].ge(100)].head(3)
        if not priority_states.empty:
            priority_text = ", ".join(
                f"{row['customer_state']} ({row['late_rate_pct']:.1f}%)" for _, row in priority_states.iterrows()
            )
            st.markdown(f"**Prioritas evaluasi pada state dengan ≥100 pesanan:** {priority_text}.")
        st.caption("Ranking memakai persentase pesanan terlambat dan jumlah pesanan sebagai denominator. Realisasi/estimasi menggunakan hari kalender; data mengidentifikasi state pelanggan.")
        st.dataframe(
            delivery_state_df.rename(columns={
                "customer_state": "State", "total_orders": "Pesanan Unik",
                "late_orders": "Pesanan Terlambat", "late_rate_pct": "Terlambat (%)",
                "avg_delivery_days": "Rata-rata Realisasi (Hari)",
                "avg_estimated_days": "Rata-rata Estimasi (Hari)",
                "avg_delay_days": "Rata-rata Selisih (Hari)",
            }).style.format({
                "Terlambat (%)": "{:.1f}", "Rata-rata Realisasi (Hari)": "{:.2f}",
                "Rata-rata Estimasi (Hari)": "{:.2f}", "Rata-rata Selisih (Hari)": "{:.2f}",
            }),
            hide_index=True,
        )

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
            ax.grid(axis="y", visible=False)
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
            ax.grid(axis="y", visible=False)
            for i, val in enumerate(seg_mon["avg_monetary"]):
                ax.text(val + seg_mon["avg_monetary"].max() * 0.02, i, f"R$ {val:,.1f}", va="center", fontsize=8)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        st.caption(
            "Biru tua menandai nilai tertinggi pada masing-masing grafik. Monetary = harga produk + ongkir. "
            "R: ≤30 / 31–90 / 91–180 / >180 hari; F: 1 / 2 / ≥3 pesanan; M: kuartil dengan penanganan nilai seri. "
            "Kohort <4 pelanggan atau belanja seragam memakai skor M netral 2. Snapshot adalah satu hari kalender setelah pembelian terakhir pada filter aktif."
        )

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
            ax.grid(axis="y", visible=False)
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


if __name__ == "__main__":
    main()

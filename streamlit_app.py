import streamlit as st
# --- FIX: Mengubah path impor agar sesuai dengan nama folder baru ---
import app_pages.beranda as beranda
import app_pages.profiling as profiling
import app_pages.segmentation as segmentation
import app_pages.behavior as behavior

# Konfigurasi halaman utama
st.set_page_config(
    page_title="SILASTIK BPS",
    page_icon="📊",
    layout="wide"
)

# --- Sidebar untuk Navigasi ---
with st.sidebar:
    st.title("📊 SILASTIK BPS")
    st.write("Dashboard Web Usage Mining")
    
    page = st.radio(
        "Pilih Halaman Analisis:",
        ("Beranda", "Pemrofilan Pengguna", "Segmentasi Pengguna", "Pola Perilaku"),
        key="page_selection"
    )
    st.info("Aplikasi ini menganalisis data log untuk menemukan pola perilaku pengguna SILASTIK BPS")

# --- Routing Halaman ---
if page == "Beranda":
    beranda.render_page()
elif page == "Pemrofilan Pengguna":
    profiling.render_page()
elif page == "Segmentasi Pengguna":
    segmentation.render_page()
elif page == "Pola Perilaku":
    behavior.render_page()

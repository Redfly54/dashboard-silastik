import streamlit as st

def render_page():
    st.title("Selamat Datang di Dashboard Web Usage Mining SILASTIK BPS")
    st.markdown(
        """
        Dashboard ini merupakan hasil dari penelitian Web Usage Mining terhadap data log akses pengguna 
        Sistem Informasi Layanan Statistik (SILASTIK) BPS.

        Tujuan dari dashboard ini adalah untuk menyajikan wawasan yang didapatkan dari tiga analisis utama:
        """
    )
    
    st.subheader("Pemrofilan Pengguna")
    st.write("Menganalisis karakteristik demografis dan teknis pengguna, serta aktivitas akses dengan website SILASTIK BPS.")

    st.subheader("Segmentasi Pengguna")
    st.write("Mengelompokkan pengguna ke dalam segmen-segmen tertentu berdasarkan perilaku navigasi mereka menggunakan algoritma K-Means clustering.")

    st.subheader("Analisis Pola Perilaku")
    st.write("Mengidentifikasi aturan asosiasi (association rules) untuk menemukan pola kunjungan halaman yang sering terjadi bersamaan pada setiap segmen pengguna.")
    
    st.info("Gunakan menu di sebelah kiri untuk menavigasi ke setiap bagian analisis.")
import streamlit as st
import plotly.express as px
import pandas as pd
from data_loader import load_preprocessed_data

def render_page():
    st.title("👤 Pemrofilan Pengguna")

    # Memuat data
    df_final, _, df_location = load_preprocessed_data()

    if df_final.empty or df_location.empty:
        st.error("Gagal memuat data. Pastikan file 'df_final.csv' dan 'hasil_lokasi_ip.csv' ada di folder 'data/'.")
        return

    # --- PERSIAPAN DATA ---
    if 'IP_Address' in df_final.columns:
        df_final.rename(columns={'IP_Address': 'IP'}, inplace=True)
    if 'IP_Address' in df_location.columns:
        df_location.rename(columns={'IP_Address': 'IP'}, inplace=True)

    df_final = pd.merge(df_final, df_location.drop_duplicates(subset=['IP']), on='IP', how='left')
    df_final['User_ID'] = df_final['IP'] + "_" + df_final['User-Agent']
    df_final['Timestamp'] = pd.to_datetime(df_final['Timestamp'])

    def get_browser(ua):
        ua_str = str(ua)
        if 'Chrome' in ua_str and 'Edge' not in ua_str: return 'Chrome'
        elif 'Firefox' in ua_str: return 'Firefox'
        elif 'Safari' in ua_str and 'Chrome' not in ua_str: return 'Safari'
        elif 'Edge' in ua_str: return 'Edge'
        else: return 'Lainnya'

    def get_os(ua):
        ua_str = str(ua)
        if 'Android' in ua_str: return 'Android'
        elif 'Windows' in ua_str: return 'Windows'
        elif 'Mac OS X' in ua_str or 'Macintosh' in ua_str: return 'Mac'
        elif 'Linux' in ua_str: return 'Linux'
        elif 'iPhone' in ua_str or 'iPad' in ua_str: return 'iOS'
        else: return 'Lainnya'

    df_final['Browser'] = df_final['User-Agent'].apply(get_browser)
    df_final['OS'] = df_final['User-Agent'].apply(get_os)
    
    # --- VISUALISASI ---
    st.header("Sebaran Geografis Pengguna")
    df_unique_users = df_final.drop_duplicates(subset=['User_ID'])
    fig_map = px.scatter_map(
        df_unique_users, lat="lat", lon="lon", hover_name="city",
        color_discrete_sequence=["#007bff"], zoom=3.5, height=500
    )
    fig_map.update_layout(mapbox_style="carto-positron", margin={"r":0,"t":40,"l":0,"b":0})
    st.plotly_chart(fig_map, use_container_width=True)

    st.header("Analisis Aktivitas Pengguna")
    df_final['Hour'] = df_final['Timestamp'].dt.hour
    hourly_counts = df_final['Hour'].value_counts().sort_index()
    fig_hour = px.line(x=hourly_counts.index, y=hourly_counts.values, markers=True, labels={'x': 'Jam (0-23)', 'y': 'Jumlah Request'}, title='Aktivitas per Jam')
    st.plotly_chart(fig_hour, use_container_width=True)

    df_final['Day'] = df_final['Timestamp'].dt.day_name()
    daily_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    daily_counts = df_final['Day'].value_counts().reindex(daily_order)
    fig_day = px.line(x=daily_counts.index, y=daily_counts.values, markers=True, labels={'x': 'Hari', 'y': 'Jumlah Request'}, title='Aktivitas per Hari')
    st.plotly_chart(fig_day, use_container_width=True)

    st.header("Analisis Perangkat Pengguna")
    col1, col2 = st.columns(2)
    with col1:
        browser_counts = df_final['Browser'].value_counts()
        fig_browser = px.bar(y=browser_counts.index, x=browser_counts.values, orientation='h', title='Distribusi Browser').update_layout(yaxis={'categoryorder':'total ascending'}, yaxis_title='Jenis Browser', xaxis_title='Jumlah Pengguna')
        st.plotly_chart(fig_browser, use_container_width=True)
    with col2:
        os_counts = df_final['OS'].value_counts()
        fig_os = px.bar(y=os_counts.index, x=os_counts.values, orientation='h', title='Distribusi Sistem Operasi').update_layout(yaxis={'categoryorder':'total ascending'}, yaxis_title='Jenis Sistem Operasi', xaxis_title='Jumlah Pengguna')
        st.plotly_chart(fig_os, use_container_width=True)

    # --- FIX: Menambahkan kembali visualisasi Sumber Akses ---
    st.header("Sumber Akses Pengguna")
    def categorize_referrer(ref):
        if 'google' in str(ref): return 'Google'
        elif 'direct access' in str(ref): return 'Akses Langsung'
        else: return 'Lainnya'
    df_final['ReferrerCategory'] = df_final['Referrer'].apply(categorize_referrer)
    referrer_counts = df_final['ReferrerCategory'].value_counts()
    
    fig_referrer = px.bar(
        y=referrer_counts.index, 
        x=referrer_counts.values, 
        orientation='h', 
        title='Distribusi Sumber Akses'
    ).update_layout(yaxis={'categoryorder':'total ascending'}, yaxis_title='Jenis Sumber Akses', xaxis_title='Jumlah Request')
    st.plotly_chart(fig_referrer, use_container_width=True)
import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample
from data_loader import load_preprocessed_data

@st.cache_data
def get_evaluation_metrics(_normalized_matrix):
    inertia, silhouette_scores, db_scores = [], [], []
    K_range = range(2, 11)
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(_normalized_matrix)
        inertia.append(kmeans.inertia_)
        db_scores.append(davies_bouldin_score(_normalized_matrix, labels))
        sample_size = 5000
        if len(_normalized_matrix) > sample_size:
            X_sample, labels_sample = resample(_normalized_matrix, labels, n_samples=sample_size, random_state=42, stratify=labels)
            score = silhouette_score(X_sample, labels_sample)
        else:
            score = silhouette_score(_normalized_matrix, labels)
        silhouette_scores.append(score)
    return pd.DataFrame({'k': K_range, 'inertia': inertia, 'silhouette_score': silhouette_scores, 'db_score': db_scores})

def render_page():
    st.title("🧩 Segmentasi Pengguna")
    _, weighted_matrix, _ = load_preprocessed_data()

    if weighted_matrix.empty:
        st.error("Gagal memuat data. Pastikan file 'weighted_matrix.csv' ada di folder 'data/'.")
        return

    scaler = MinMaxScaler()
    normalized_matrix = pd.DataFrame(scaler.fit_transform(weighted_matrix), index=weighted_matrix.index, columns=weighted_matrix.columns)
    
    st.header("Evaluasi Model Clustering")
    evaluation_df = get_evaluation_metrics(normalized_matrix)

    col1, col2 = st.columns(2)
    with col1:
        fig_elbow = px.line(evaluation_df, x='k', y='inertia', title='Metode Elbow', markers=True).update_layout(xaxis_title='Jumlah Cluster (k)', yaxis_title='Inertia')
        st.plotly_chart(fig_elbow, use_container_width=True)
    with col2:
        fig_silhouette = px.line(evaluation_df, x='k', y='silhouette_score', title='Skor Silhouette (Higher is Better)', markers=True).update_layout(xaxis_title='Jumlah Cluster (k)', yaxis_title='Silhouette Score')
        st.plotly_chart(fig_silhouette, use_container_width=True)

    # --- FIX: Menambahkan tabel Davies-Bouldin Index ---
    st.subheader("Tabel Davies-Bouldin Index (Lower is Better)")
    db_df = evaluation_df[['k', 'db_score']].copy()
    db_df['db_score'] = db_df['db_score'].round(3)
    db_df = db_df.rename(columns={'k': 'Jumlah Cluster (k)', 'db_score': 'Davies-Bouldin Index'})
    st.table(db_df.set_index('Jumlah Cluster (k)'))


    st.header("Hasil Segmentasi Interaktif")
    k_value = st.slider("Pilih Jumlah Cluster (k)", min_value=2, max_value=10, value=3, step=1)

    kmeans = KMeans(n_clusters=k_value, random_state=42, n_init=10)
    labels = kmeans.fit_predict(normalized_matrix)
    
    pca = PCA(n_components=2)
    matrix_pca = pca.fit_transform(normalized_matrix)
    
    df_plot = normalized_matrix.reset_index()
    df_plot['cluster'] = 'Segmen ' + pd.Series(labels).astype(str)
    df_plot['pca1'] = matrix_pca[:, 0]
    df_plot['pca2'] = matrix_pca[:, 1]
    
    fig_pca = px.scatter(df_plot, x='pca1', y='pca2', color='cluster', hover_data=['User_ID', 'Session_ID'], title=f'Visualisasi Segmen (k={k_value})')
    st.plotly_chart(fig_pca, use_container_width=True)
    
    # --- FIX: Menambahkan metrik hasil untuk k yang dipilih ---
    st.subheader(f"Metrik Evaluasi untuk k = {k_value}")
    
    db_score = davies_bouldin_score(normalized_matrix, labels)
    
    sample_size = 5000
    if len(normalized_matrix) > sample_size:
        X_sample, labels_sample = resample(normalized_matrix, labels, n_samples=sample_size, random_state=42, stratify=labels)
        silhouette_val = silhouette_score(X_sample, labels_sample)
        st.metric(label="Skor Silhouette (diestimasi dari sampel)", value=f"{silhouette_val:.3f}")
    else:
        silhouette_val = silhouette_score(normalized_matrix, labels)
        st.metric(label="Skor Silhouette", value=f"{silhouette_val:.3f}")
        
    st.metric(label="Davies-Bouldin Index", value=f"{db_score:.3f}")
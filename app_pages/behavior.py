import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from mlxtend.frequent_patterns import fpgrowth, association_rules
import networkx as nx
from data_loader import load_preprocessed_data

def create_network_graph(rules_df):
    if rules_df.empty:
        return go.Figure(layout=go.Layout(title="Tidak ada aturan untuk divisualisasikan"))
    
    rules_df['antecedents_str'] = rules_df['antecedents'].apply(lambda x: ', '.join(list(x)))
    rules_df['consequents_str'] = rules_df['consequents'].apply(lambda x: ', '.join(list(x)))
    G = nx.from_pandas_edgelist(rules_df, 'antecedents_str', 'consequents_str', ['lift'])
    pos = nx.spring_layout(G, k=0.8, iterations=50, seed=42)
    
    # --- FIX: Inisialisasi variabel baris per baris ---
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]; x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None]); edge_y.extend([y0, y1, None])
        
    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=1, color='#888'), hoverinfo='none', mode='lines')
    
    # --- FIX: Inisialisasi variabel baris per baris ---
    node_x = []
    node_y = []
    node_text = []
    node_size = []
    for node in G.nodes():
        x, y = pos[node]; node_x.append(x); node_y.append(y); node_text.append(node)
        node_size.append(15 + 10 * (rules_df['antecedents_str'].str.contains(node, regex=False) | rules_df['consequents_str'].str.contains(node, regex=False)).sum())
        
    node_trace = go.Scatter(
        x=node_x, y=node_y, mode='markers+text', hoverinfo='text', text=node_text, textposition="top center",
        marker=dict(
            showscale=True, colorscale='YlGnBu', size=node_size, line_width=2,
            colorbar=dict(thickness=15, title='Koneksi Node', xanchor='left', title_side='right')
        )
    )
    
    node_adjacencies = [len(adj) for adj in G.adj.values()]
    node_trace.marker.color = node_adjacencies
    node_trace.hovertext = [f"{text}<br># koneksi: {adj}" for text, adj in zip(node_text, node_adjacencies)]
    
    fig = go.Figure(data=[edge_trace, node_trace],
             layout=go.Layout(
                title='Grafik Aturan Asosiasi Antar Kategori', showlegend=False, hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
            )
    return fig

def render_page():
    st.title("🔄 Pola Perilaku Pengguna")
    _, weighted_matrix, _ = load_preprocessed_data()

    if weighted_matrix.empty:
        st.error("Gagal memuat data.")
        return

    scaler = MinMaxScaler()
    normalized_matrix = pd.DataFrame(scaler.fit_transform(weighted_matrix), index=weighted_matrix.index, columns=weighted_matrix.columns)

    st.header("Analisis Aturan Asosiasi per Segmen")
    
    col1, col2 = st.columns(2)
    with col1:
        k_value = st.slider("1. Pilih Jumlah Cluster (k)", 2, 10, 3)
    with col2:
        selected_cluster = st.selectbox("2. Pilih Segmen", range(k_value))

    min_support = st.slider("Minimum Support", 0.05, 0.5, 0.15)
    min_confidence = st.slider("Minimum Confidence", 0.1, 1.0, 0.7)
    min_lift = st.slider("Minimum Lift (untuk Grafik)", 1.0, 5.0, 1.4)


    kmeans = KMeans(n_clusters=k_value, random_state=42, n_init=10)
    labels = kmeans.fit_predict(normalized_matrix)
    
    matrix_with_clusters = weighted_matrix.copy()
    matrix_with_clusters['Cluster'] = labels
    cluster_data = matrix_with_clusters[matrix_with_clusters['Cluster'] == selected_cluster].drop('Cluster', axis=1)
    
    cluster_data_binary = (cluster_data > 0)
    frequent_itemsets = fpgrowth(cluster_data_binary, min_support=min_support, use_colnames=True)
    
    if not frequent_itemsets.empty:
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
        if not rules.empty:
            st.subheader(f"Aturan Asosiasi untuk Segmen {selected_cluster}")
            rules_display = rules.copy()
            rules_display['antecedents'] = rules_display['antecedents'].apply(lambda x: ', '.join(list(x)))
            rules_display['consequents'] = rules_display['consequents'].apply(lambda x: ', '.join(list(x)))
            st.dataframe(rules_display[['antecedents', 'consequents', 'support', 'confidence', 'lift']].round(3))

            st.subheader(f"Grafik Jaringan Aturan (Lift ≥ {min_lift})")
            rules_for_graph = rules[rules['lift'] >= min_lift]
            if rules_for_graph.empty:
                st.warning(f"Tidak ada aturan yang memenuhi syarat visualisasi (Lift ≥ {min_lift}). Coba turunkan nilai minimum lift.")
            else:
                fig_network = create_network_graph(rules_for_graph)
                st.plotly_chart(fig_network, use_container_width=True)

        else:
            st.warning("Tidak ada aturan asosiasi yang terbentuk dengan parameter ini.")
    else:
        st.warning("Tidak ditemukan itemset yang sering muncul dengan parameter ini.")

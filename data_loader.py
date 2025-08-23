import pandas as pd
import sys
import os

# --- KONFIGURASI PATH DATA YANG SUDAH DIOLAH ---
# Menentukan path absolut untuk keandalan
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')

FINAL_DATA_PATH = os.path.join(DATA_DIR, 'df_final.csv')
WEIGHTED_MATRIX_PATH = os.path.join(DATA_DIR, 'weighted_matrix.csv')
LOCATION_DATA_PATH = os.path.join(DATA_DIR, 'hasil_lokasi_ip.csv') # Path untuk file lokasi

def load_preprocessed_data():
    """
    Memuat semua data yang sudah diproses untuk analisis.

    Returns:
        tuple: Sebuah tuple berisi tiga DataFrame:
               - df_final (pd.DataFrame): Data utama untuk pemrofilan.
               - weighted_matrix (pd.DataFrame): Matriks untuk pemodelan.
               - df_location (pd.DataFrame): Data lokasi IP.
    """
    print("Mencoba memuat data yang sudah diproses...")
    
    try:
        # Memeriksa keberadaan semua file yang dibutuhkan
        for path in [FINAL_DATA_PATH, WEIGHTED_MATRIX_PATH, LOCATION_DATA_PATH]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"File tidak ditemukan di path: {path}")

        df_final = pd.read_csv(FINAL_DATA_PATH)
        weighted_matrix = pd.read_csv(WEIGHTED_MATRIX_PATH, index_col=[0, 1])
        df_location = pd.read_csv(LOCATION_DATA_PATH)
        
        print("✅ Semua data berhasil dimuat.")
        return df_final, weighted_matrix, df_location
        
    except Exception as e:
        print(f"❌ ERROR saat memuat data: {e}")
        print("Pastikan file 'df_final.csv', 'weighted_matrix.csv', dan 'hasil_lokasi_ip.csv' berada di dalam folder 'data/'.")
        # Mengembalikan DataFrame kosong agar aplikasi tidak crash
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

if __name__ == '__main__':
    df, matrix, locations = load_preprocessed_data()
    if not df.empty:
        print("\n--- Pengecekan Data Berhasil ---")
        print("\nContoh data pemrofilan (df_final):")
        print(df.head())
        print("\nContoh data lokasi (hasil_lokasi_ip):")
        print(locations.head())

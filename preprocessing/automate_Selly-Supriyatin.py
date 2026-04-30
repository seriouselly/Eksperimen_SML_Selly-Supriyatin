import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def run_preprocessing(file_path):
    print("Membaca data dari:", file_path)
    df = pd.read_csv(file_path)
    
    X = df.drop(columns=['math score'])
    y = df['math score']
    
    X_train, _, _, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    
    numeric_features = ['reading score', 'writing score']
    categorical_features = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first'), categorical_features)
        ])
    
    print("Membersihkan dan mengubah data...")
    X_train_processed = preprocessor.fit_transform(X_train)
    
    # Mengubah hasil yang berupa matriks angka kembali menjadi tabel DataFrame
    # Lalu menyimpannya menjadi file CSV baru agar bisa dilihat hasilnya
    df_processed = pd.DataFrame(X_train_processed)
    output_path = os.path.join(os.path.dirname(__file__), 'dataset_processed.csv')
    df_processed.to_csv(output_path, index=False)
    print("Berhasil! Data bersih disimpan sebagai 'dataset_processed.csv'")

if __name__ == "__main__":
    # Tentukan path yang benar tergantung di mana script dijalankan
    paths_to_try = [
        'dataset_raw/StudentsPerformance.csv',  # Jika dijalankan dari root
        '../dataset_raw/StudentsPerformance.csv'  # Jika dijalankan dari preprocessing/
    ]
    
    file_path = None
    for path in paths_to_try:
        if os.path.exists(path):
            file_path = path
            break
    
    if file_path is None:
        raise FileNotFoundError("StudentsPerformance.csv tidak ditemukan di mana pun!")
    
    run_preprocessing(file_path)
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import os

def run_preprocessing(raw_data_path='/StudentsPerformance_raw/StudentsPerformance.csv',
                      output_dir='./StudentsPerformance_processing'):
    # Baca data
    df = pd.read_csv(raw_data_path)
    
    # Encoding
    categorical_cols = ['gender', 'race/ethnicity', 'parental level of education', 
                        'lunch', 'test preparation course']
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    
    # Fitur dan target
    X = df.drop('math score', axis=1)
    y = df['math score']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Normalisasi
    numeric_features = ['reading score', 'writing score']
    scaler = StandardScaler()
    X_train[numeric_features] = scaler.fit_transform(X_train[numeric_features])
    X_test[numeric_features] = scaler.transform(X_test[numeric_features])
    
    # Simpan
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump((X_train, X_test, y_train, y_test), f'{output_dir}/processed_data.pkl')
    joblib.dump(scaler, f'{output_dir}/scaler.pkl')
    joblib.dump(label_encoders, f'{output_dir}/label_encoders.pkl')
    print(f"Preprocessing selesai. Data tersimpan di {output_dir}")

if __name__ == "__main__":
    run_preprocessing()
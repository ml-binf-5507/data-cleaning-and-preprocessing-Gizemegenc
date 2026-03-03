import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print("Libraries loaded!")
import os
import sys

# Set repo root
repo_root = r"C:\Users\gizem\Desktop\Humber College\Semester 2\Machine Learning AI Bioinforma\BINF-5507-Winter2026\BINF-5507-Winter2026\Assignment_1_GizemEsraGenc"
os.chdir(repo_root)
sys.path.append(os.path.join(repo_root, "src"))

# Import your functions
from preprocess import detect_feature_types, encode_categorical, scale_numeric
print("✅ Functions imported successfully!")


import pandas as pd

# Load your heart disease dataset
df = pd.read_csv("heart_disease_dataset.csv")

# 1️⃣ Detect feature types
cat_cols, num_cols = detect_feature_types(df)
print("Categorical columns:", cat_cols)
print("Numeric columns:", num_cols)

# 2️⃣ Encode categorical features
df_encoded, new_cols = encode_categorical(df, cat_cols)
print("New one-hot encoded columns added:", new_cols)

# 3️⃣ Scale numeric features
df_scaled = scale_numeric(df_encoded, num_cols)
print("Preview of scaled dataframe:")
print(df_scaled.head())
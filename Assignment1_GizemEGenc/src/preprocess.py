import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print("Libraries loaded!")
def load_data("C:\Users\gizem\Desktop\Humber College\Semester 2\Machine Learning AI Bioinforma\BINF-5507-Winter2026\BINF-5507-Winter2026\Assignment_1_GizemEsraGenc\heart_disease_dataset.csv"):
    """
    Load the dataset from a CSV file.
    
    Parameters:
    file_path (str): The path to the CSV file containing the dataset.
    
    Returns:
    pd.DataFrame: The loaded dataset as a pandas DataFrame.
    """
    try:
        data = pd.read_csv(file_path)
        print(f"Data loaded successfully from {file_path}")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
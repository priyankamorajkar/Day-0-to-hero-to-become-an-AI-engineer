import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import tkinter as tk
from tkinter import filedialog
from sklearn.linear_model import LinearRegression

class CSVInsightGenerator:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None

    def load_and_clean(self):
        print(f"\n[1/3] Loading: {os.path.basename(self.file_path)}...")
        try:
            self.df = pd.read_csv(self.file_path, sep=None, engine='python')
            self.df.columns = self.df.columns.str.strip()

            for col in self.df.columns:
                if self.df[col].dtype == 'object':
                    converted = pd.to_numeric(self.df[col].replace(r'[^0-9.-]', '', regex=True), errors='coerce')
                    if not converted.isna().all():
                        self.df[col] = converted

            self.df = self.df.dropna(axis=1, how='all')
            numeric_df = self.df.select_dtypes(include=[np.number])

            if not numeric_df.empty:
                print(f"SUCCESS: {self.df.shape[0]} rows found.\n--- Data Summary ---")
                print(numeric_df.describe().round(2))
                return True
            
            print("ERROR: No numeric columns found.")
            return False
        except Exception as e:
            print(f"LOAD ERROR: {e}")
            return False

    def generate_visuals(self):
        print("\n[2/3] Generating Visuals...")
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            plt.figure(figsize=(10, 5))
            sns.histplot(self.df[numeric_cols[0]].dropna(), kde=True, color='teal')
            plt.title(f'Distribution Analysis: {numeric_cols[0]}')
            plt.show() 

    def detect_trends(self):
        print("\n[3/3] Running ML Trend Detection...")
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) >= 1:
            y = self.df[numeric_cols[0]].dropna().values
            if len(y) > 1:
                X = np.arange(len(y)).reshape(-1, 1)
                model = LinearRegression().fit(X, y)
                slope = model.coef_[0]
                print(f"RESULT: {'UPWARD' if slope > 0 else 'DOWNWARD'} trend (Slope: {slope:.4f})")

def get_path():
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.askopenfilename(title="Select CSV File", filetypes=[("CSV Files", "*.csv")])
    root.destroy()
    return path

def main():
    print("--- CSV Insight Generator ---")
    path = get_path()
    if path:
        gen = CSVInsightGenerator(path)
        if gen.load_and_clean():
            gen.generate_visuals()
            gen.detect_trends()
        print("\nPROCESS COMPLETE.")
    else:
        print("ABORTED: No file selected.")

if __name__ == "__main__":
    main()
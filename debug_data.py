import pandas as pd
import os
from src.utils.config import settings

print(f"Config Data Path: {settings.DATA_PATH}")
if os.path.exists(settings.DATA_PATH):
    print(f"File exists. Size: {os.path.getsize(settings.DATA_PATH)} bytes")
else:
    print("File DOES NOT exist.")

try:
    print("Attempting read with engine='openpyxl'...")
    df = pd.read_excel(settings.DATA_PATH, header=1, engine='openpyxl')
    print("Success with openpyxl!")
    print(df.head())
except Exception as e:
    print(f"ERROR with openpyxl: {e}")

try:
    print("Attempting read with default engine...")
    df = pd.read_excel(settings.DATA_PATH, header=1)
    print("Success with default!")
except Exception as e:
    print(f"ERROR with default: {e}")

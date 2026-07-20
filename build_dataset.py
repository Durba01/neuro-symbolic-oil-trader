import yfinance as yf
import pandas as pd
import os

def download_oil_data():
    print("Downloading WTI Crude Oil Historical Data...")
    
    # CL=F is the global ticker symbol for WTI Crude Oil Futures
    oil_data = yf.download("CL=F", period="2y", interval="1d")
    
    if oil_data.empty:
        print("Failed to download data.")
        return
        
    # Clean up the data
    oil_data.reset_index(inplace=True)
    
    # Save it to a CSV file in our project folder
    csv_filename = "wti_historical_data.csv"
    oil_data.to_csv(csv_filename, index=False)
    
    print(f"✅ Success! Saved {len(oil_data)} days of WTI Crude Oil history to {csv_filename}")
    print("\nHere is a preview of what the Neural Network will study:")
    print(oil_data.tail())

if __name__ == "__main__":
    download_oil_data()
import yfinance as yf

df = yf.download(
    "RELIANCE.NS",
    start="2020-01-01",
    end="2025-01-01"
)

print(df.head())
print("\nShape:", df.shape)

df.to_csv("reliance.csv")

print("\nData Saved Successfully!")
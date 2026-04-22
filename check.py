import pandas as pd

trader_df    = pd.read_csv("historical_data.csv")
sentiment_df = pd.read_csv("fear_greed_index.csv")

trader_df.columns    = trader_df.columns.str.strip().str.lower().str.replace(' ', '_')
sentiment_df.columns = sentiment_df.columns.str.strip().str.lower().str.replace(' ', '_')

print("TRADER COLUMNS")
print(trader_df.columns.tolist())

print("\nSENTIMENT COLUMNS")
print(sentiment_df.columns.tolist())

print("\nFIRST TIMESTAMP VALUE")
print(repr(trader_df['timestamp_ist'].iloc[0]))

print("\nFIRST SENTIMENT DATE")
print(repr(sentiment_df['date'].iloc[0]))

print("\nTRADER SHAPE")
print(trader_df.shape)

print("\nSENTIMENT SHAPE")
print(sentiment_df.shape)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 5)

# load both datasets
trader_df = pd.read_csv("historical_data.csv")
sentiment_df = pd.read_csv("fear_greed_index.csv")

trader_df.columns = trader_df.columns.str.strip().str.lower().str.replace(' ', '_')
sentiment_df.columns = sentiment_df.columns.str.strip().str.lower().str.replace(' ', '_')

# trader timestamps are DD-MM-YYYY HH:MM, sentiment is YYYY-MM-DD
trader_df['date'] = pd.to_datetime(
    trader_df['timestamp_ist'], format='%d-%m-%Y %H:%M', errors='coerce'
).dt.normalize()

sentiment_df['date'] = pd.to_datetime(
    sentiment_df['date'], format='%Y-%m-%d', errors='coerce'
).dt.normalize()

# merge on date so each trade gets the sentiment for that day
sentiment_slim = sentiment_df[['date', 'classification', 'value']].copy()
merged = pd.merge(trader_df, sentiment_slim, on='date', how='left')

# clean up numeric columns
merged['closed_pnl'] = pd.to_numeric(merged['closed_pnl'], errors='coerce')
merged['size_usd'] = pd.to_numeric(merged['size_usd'], errors='coerce')
merged['execution_price'] = pd.to_numeric(merged['execution_price'], errors='coerce')

# only keep rows that actually have a closed trade and a sentiment label
closed = merged[
    merged['closed_pnl'].notna() &
    (merged['closed_pnl'] != 0) &
    merged['classification'].notna()
].copy()

closed['is_win'] = closed['closed_pnl'] > 0

# consistent left-to-right ordering: fear → greed
ALL_SENTIMENTS = ['Extreme Fear', 'Fear', 'Neutral', 'Greed', 'Extreme Greed']
SENTIMENT_ORDER = [s for s in ALL_SENTIMENTS if s in closed['classification'].unique()]


# avg pnl per sentiment — do traders make more during greed?
pnl_summary = (
    closed.groupby('classification')['closed_pnl']
    .agg(['mean', 'median', 'count'])
    .reindex(SENTIMENT_ORDER)
    .rename(columns={'mean': 'Avg PnL', 'median': 'Median PnL', 'count': 'Trades'})
)

fig, ax = plt.subplots()
colors = ['#e74c3c' if v < 0 else '#2ecc71' for v in pnl_summary['Avg PnL']]
bars = ax.bar(pnl_summary.index, pnl_summary['Avg PnL'], color=colors, edgecolor='white', width=0.6)
ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
ax.set_title('Average Closed PnL by Market Sentiment', fontsize=14, fontweight='bold')
ax.set_xlabel('Market Sentiment')
ax.set_ylabel('Average PnL (USD)')
plt.xticks(rotation=30, ha='right')
for bar, val in zip(bars, pnl_summary['Avg PnL']):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + (0.5 if val >= 0 else -1.5),
            f'${val:.2f}', ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.savefig('chart1_avg_pnl_by_sentiment.png', dpi=150)
plt.show()


# trade volume per sentiment — when are traders most active?
trade_counts = closed['classification'].value_counts().reindex(SENTIMENT_ORDER)

fig, ax = plt.subplots()
ax.bar(trade_counts.index, trade_counts.values, color='steelblue', edgecolor='white', width=0.6)
ax.set_title('Number of Trades by Market Sentiment', fontsize=14, fontweight='bold')
ax.set_xlabel('Market Sentiment')
ax.set_ylabel('Number of Trades')
plt.xticks(rotation=30, ha='right')
for i, val in enumerate(trade_counts.values):
    ax.text(i, val + 50, f'{int(val):,}', ha='center', fontsize=9)
plt.tight_layout()
plt.savefig('chart2_trade_count_by_sentiment.png', dpi=150)
plt.show()


# box plot — trim extreme outliers so the chart is readable
q_low = closed['closed_pnl'].quantile(0.02)
q_high = closed['closed_pnl'].quantile(0.98)
plot_df = closed[(closed['closed_pnl'] >= q_low) & (closed['closed_pnl'] <= q_high)]

fig, ax = plt.subplots()
sns.boxplot(data=plot_df, x='classification', y='closed_pnl',
            order=SENTIMENT_ORDER, palette='RdYlGn', ax=ax)
ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
ax.set_title('PnL Distribution by Market Sentiment', fontsize=14, fontweight='bold')
ax.set_xlabel('Market Sentiment')
ax.set_ylabel('Closed PnL (USD)')
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.savefig('chart3_pnl_distribution_boxplot.png', dpi=150)
plt.show()


# win rate — what % of trades close green per sentiment?
win_rate = (
    closed.groupby('classification')['is_win']
    .mean().mul(100).reindex(SENTIMENT_ORDER)
)

fig, ax = plt.subplots()
bars = ax.bar(win_rate.index, win_rate.values, color='mediumseagreen', edgecolor='white', width=0.6)
ax.axhline(50, color='gray', linewidth=1, linestyle='--', label='50% breakeven line')
ax.set_title('Win Rate (%) by Market Sentiment', fontsize=14, fontweight='bold')
ax.set_xlabel('Market Sentiment')
ax.set_ylabel('Win Rate (%)')
ax.set_ylim(0, 100)
ax.legend()
plt.xticks(rotation=30, ha='right')
for bar, val in zip(bars, win_rate.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'{val:.1f}%', ha='center', fontsize=9)
plt.tight_layout()
plt.savefig('chart4_win_rate_by_sentiment.png', dpi=150)
plt.show()


# buy vs sell breakdown — do traders go long more during greed?
side_sentiment = (
    closed.groupby(['classification', 'side'])
    .size().unstack(fill_value=0).reindex(SENTIMENT_ORDER)
)

fig, ax = plt.subplots()
side_sentiment.plot(kind='bar', colormap='Set2', edgecolor='white', width=0.6, ax=ax)
ax.set_title('Buy vs Sell Count by Market Sentiment', fontsize=14, fontweight='bold')
ax.set_xlabel('Market Sentiment')
ax.set_ylabel('Number of Trades')
ax.legend(title='Side')
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.savefig('chart5_buy_sell_by_sentiment.png', dpi=150)
plt.show()


# avg position size — do traders risk more capital during greed?
trade_size = closed.groupby('classification')['size_usd'].mean().reindex(SENTIMENT_ORDER)

fig, ax = plt.subplots()
bars = ax.bar(trade_size.index, trade_size.values, color='mediumpurple', edgecolor='white', width=0.6)
ax.set_title('Average Trade Size (USD) by Market Sentiment', fontsize=14, fontweight='bold')
ax.set_xlabel('Market Sentiment')
ax.set_ylabel('Avg Trade Size (USD)')
plt.xticks(rotation=30, ha='right')
for bar, val in zip(bars, trade_size.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
            f'${val:,.0f}', ha='center', fontsize=9)
plt.tight_layout()
plt.savefig('chart6_trade_size_by_sentiment.png', dpi=150)
plt.show()


# summary table across all sentiment buckets
summary = (
    closed.groupby('classification').agg(
        Total_Trades  = ('closed_pnl', 'count'),
        Avg_PnL       = ('closed_pnl', 'mean'),
        Median_PnL    = ('closed_pnl', 'median'),
        Total_PnL     = ('closed_pnl', 'sum'),
        Win_Rate_Pct  = ('is_win', lambda x: round(x.mean() * 100, 1)),
        Avg_Trade_USD = ('size_usd', 'mean'),
    )
    .reindex(SENTIMENT_ORDER)
    .round(2)
)
print(summary.to_string())
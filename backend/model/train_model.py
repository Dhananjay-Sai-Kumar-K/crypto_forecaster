import pandas as pd
from prophet import Prophet
import joblib
import os
import sys

print("CWD:", os.getcwd())


def main():
	# Compute repo root relative to this script so paths are robust
	here = os.path.dirname(os.path.abspath(__file__))
	repo_root = os.path.abspath(os.path.join(here, '..', '..'))

	# === Load Data ===
	data_path = os.path.join(repo_root, 'data', 'cryptocurrency.csv')
	if not os.path.exists(data_path):
		print(f"Error: data file not found at {data_path}")
		print(f"Make sure you run this script from the project or that the file exists.")
		sys.exit(1)

	df = pd.read_csv(data_path)

	# === Basic Cleaning ===
	if 'timestamp' not in df.columns or 'price_usd' not in df.columns or 'name' not in df.columns:
		print("Error: expected columns 'timestamp', 'price_usd', and 'name' in the CSV.")
		print(f"Found columns: {list(df.columns)}")
		sys.exit(1)

	df['timestamp'] = pd.to_datetime(df['timestamp'])
	df = df.sort_values('timestamp')

	# Select one coin (e.g., Bitcoin)
	coin_name = 'Bitcoin'
	coin_df = df[df['name'] == coin_name][['timestamp', 'price_usd']].dropna()
	if coin_df.empty:
		available = ', '.join(map(str, df['name'].unique()[:20]))
		print(f"No rows found for coin '{coin_name}'. Available sample coins: {available}")
		sys.exit(1)

	# Clean numeric column: remove commas/currency and convert to float
	try:
		# Ensure values are strings first, then remove common thousands separators and currency symbols
		coin_df['price_usd'] = coin_df['price_usd'].astype(str).str.replace(',', '')
		coin_df['price_usd'] = coin_df['price_usd'].str.replace('$', '')
		coin_df['price_usd'] = coin_df['price_usd'].astype(float)
	except Exception as e:
		print("Error converting 'price_usd' to numeric:", e)
		print("Sample values:", coin_df['price_usd'].head(10).tolist())
		sys.exit(1)

	# Prophet expects columns 'ds' and 'y'
	coin_df = coin_df.rename(columns={'timestamp': 'ds', 'price_usd': 'y'})

	# === Train Prophet Model ===
	model = Prophet(daily_seasonality=True)
	model.fit(coin_df)

	# === Forecast next 30 days ===
	future = model.make_future_dataframe(periods=30)
	forecast = model.predict(future)

	# === Save Model & Forecast ===
	model_dir = os.path.join(repo_root, 'backend', 'model')
	os.makedirs(model_dir, exist_ok=True)
	model_path = os.path.join(model_dir, 'model.pkl')
	joblib.dump(model, model_path)

	forecast_path = os.path.join(repo_root, 'data', 'forecast_output.csv')
	forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv(forecast_path, index=False)

	print(f"✅ Model trained successfully and saved as {model_path}")
	print(f"✅ Forecast saved as {forecast_path}")


if __name__ == '__main__':
	main()
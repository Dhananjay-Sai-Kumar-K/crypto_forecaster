from flask import Flask, request, jsonify
import joblib
import pandas as pd
from prophet import Prophet
import os
import sys
import traceback

try:
    import mysql.connector
except Exception:
    mysql = None


app = Flask(__name__)


# Robust model loading
def load_model():
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(here, '..'))
    model_path = os.path.join(repo_root, 'model', 'model.pkl')
    if not os.path.exists(model_path):
        # Also try backend/model relative path for backward-compat
        alt = os.path.join(repo_root, 'backend', 'model', 'model.pkl')
        if os.path.exists(alt):
            model_path = alt
    if not os.path.exists(model_path):
        print(f"Model file not found at {model_path}")
        return None
    try:
        return joblib.load(model_path)
    except Exception as e:
        print(f"Failed to load model from {model_path}:", e)
        return None


model = load_model()
if model is None:
    print("Warning: Model not loaded. /predict will return an error until a model is available.")


# Database connection (configurable via environment variables)
def get_db_connection():
    if mysql is None:
        raise RuntimeError('mysql.connector not available')
    db_host = os.environ.get('DB_HOST', 'localhost')
    db_user = os.environ.get('DB_USER', 'root')
    db_pass = os.environ.get('DB_PASS', 'yourpassword')
    db_name = os.environ.get('DB_NAME', 'crypto_forecast')
    return mysql.connector.connect(host=db_host, user=db_user, password=db_pass, database=db_name)


@app.route('/')
def index():
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None
    })


@app.route('/predict', methods=['GET'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded on server'}), 500

    try:
        days = int(request.args.get('days', 7))
    except Exception:
        return jsonify({'error': 'Invalid days parameter'}), 400

    # Forecast
    future = model.make_future_dataframe(periods=days)
    forecast = model.predict(future)

    # Prepare result rows (take last `days` entries)
    rows = forecast[['ds', 'yhat']].tail(days).copy()
    rows['ds'] = rows['ds'].dt.date
    result = rows.to_dict(orient='records')

    # Attempt to save to database, but don't fail the request if DB is down
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                date DATE PRIMARY KEY,
                predicted_price FLOAT
            )
        ''')
        for r in result:
            cursor.execute('''
                INSERT INTO predictions (date, predicted_price)
                VALUES (%s, %s)
                ON DUPLICATE KEY UPDATE predicted_price=%s
            ''', (r['ds'], float(r['yhat']), float(r['yhat'])))
        conn.commit()
        conn.close()
    except Exception:
        # Log exception to console but don't crash the endpoint
        print('Database write failed:')
        traceback.print_exc()

    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True)

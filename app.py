import eventlet
eventlet.monkey_patch()  # ✅ Must be the first import
from eventlet.green import urllib

from flask import Flask, request, render_template, jsonify
from flask_socketio import SocketIO, emit
import pandas as pd
import numpy as np
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from threading import Lock
import os
import time
import json

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet", ping_interval=10, ping_timeout=20)
global_model_lock = Lock()

# Logging setup
logging.basicConfig(filename="static/app.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load dataset
data = pd.read_csv("./new.csv")
data = data.assign(Sentence=data['Sentence'].fillna(""))
data = data.dropna(subset=['Label'])

X = data['Sentence']
y = data['Label']

vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Initialize SVM classifier
svm_classifier = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-4, max_iter=1000, tol=1e-3, random_state=42)
svm_classifier.fit(X_vectorized, y)

# Store connected users
connected_users = {}

def get_geo_location(ip):
    """Get location from IP address"""
    try:
        url = f'http://ipinfo.io/{ip}/json'
        response = urllib.request.urlopen(url)
        data = json.loads(response.read().decode())
        loc = data.get('loc', '0,0').split(',')
        return {
            'lat': float(loc[0]),
            'lon': float(loc[1]),
            'location': data.get('city', 'Unknown') + ', ' + data.get('country', 'Unknown')
        }
    except Exception as e:
        logging.error(f"Error getting location for {ip}: {str(e)}")
        return {'lat': 0, 'lon': 0, 'location': 'Unknown'}

@app.route('/')
def index():
    return render_template('index.html')


# SocketIO event to handle user location
@socketio.on('user_location')
def handle_user_location(data):
    """Handles location data from users on newpage.html and results.html"""
    lat = data['lat']
    lon = data['lon']
    # Update the connected user with location data
    if request.sid not in connected_users:
        connected_users[request.sid] = {}
    connected_users[request.sid]['lat'] = lat
    connected_users[request.sid]['lon'] = lon
    logging.info(f"User {request.sid} location updated: Latitude: {lat}, Longitude: {lon}")
    
    # Emit the updated user location to all clients (admin page can use this)
    emit('update_users', connected_users, broadcast=True)

# SocketIO event to send location data to the admin page
@socketio.on('connect')
def handle_connect():
    """Handles new client connection"""
    try:
        client_ip = request.remote_addr
        location = get_geo_location(client_ip)
        connected_users[request.sid] = {
            'ip': client_ip,
            'location': location['location'],
            'lat': location['lat'],
            'lon': location['lon']
        }
        logging.info(f"New client connected: {client_ip} from {location['location']}")
        emit('update_users', connected_users, broadcast=True)
    except Exception as e:
        logging.error(f"Error during connection: {str(e)}")

@app.route('/admin')
def admin():
    return render_template('admin.html')

# Send updates to admin.html when connected users change
@socketio.on('update_users')
def send_location_to_admin(data):
    """Emit the updated connected users to admin page"""
    emit('update_location', data, broadcast=True)

@app.route('/metrics', methods=['GET'])
def get_metrics():
    """Return model performance metrics"""
    y_pred = svm_classifier.predict(X_vectorized)

    # Calculate existing metrics
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, average='weighted', zero_division=1)
    recall = recall_score(y, y_pred, average='weighted', zero_division=1)
    f1 = f1_score(y, y_pred, average='weighted', zero_division=1)

    # Calculate additional metrics
    data_coverage = len(X.dropna()) / len(X) * 100  # Coverage as percentage of non-null training data
    malicious_query_count = sum(1 for label in y if label == '0')  # Assuming '0' indicates malicious queries
    total_queries = len(y)
    malicious_query_percent = (malicious_query_count / total_queries) * 100 if total_queries > 0 else 0

    # Example of calculating query size and duration
    query_sample = "SELECT * FROM users WHERE username = 'test'"  # Example query, update this based on actual use
    query_size = len(query_sample.encode('utf-8')) / 1024  # Query size in KB
    
    # Simulate update duration (example for illustration)
    start_time = time.time()
    # Your update model code here (simulate some operation)
    time.sleep(1)  # Simulating model update time
    update_duration = time.time() - start_time

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "dataCoverage": data_coverage,
        "maliciousQueryPercent": malicious_query_percent,
        "querySize": query_size,
        "updateDuration": update_duration
    }

    return jsonify(metrics)

@app.route('/get_model', methods=['GET'])
def get_model():
    """Return current model parameters"""
    with global_model_lock:
        model_params = {
            'coef': svm_classifier.coef_.tolist(),
            'intercept': svm_classifier.intercept_.tolist(),
            'vocabulary': vectorizer.vocabulary_
        }
    return jsonify(model_params)

@app.route('/update_model', methods=['POST'])
def update_model():
    """Update model with new parameters from clients"""
    global svm_classifier
    update_data = request.get_json()
    
    client_coef = np.array(update_data['coef'])
    client_intercept = np.array(update_data['intercept'])
    
    with global_model_lock:
        # Federated averaging
        server_coef = svm_classifier.coef_
        server_intercept = svm_classifier.intercept_
        
        new_coef = (server_coef + client_coef) / 2
        new_intercept = (server_intercept + client_intercept) / 2
        
        svm_classifier.coef_ = new_coef
        svm_classifier.intercept_ = new_intercept
    
    return jsonify({'status': 'update successful'})

def detect_sql_injection(query, classifier):
    query_vectorized = vectorizer.transform([query])
    prediction = classifier.predict(query_vectorized)
    return "Malicious SQL Injection Attempt" if prediction == '0' else "Benign SQL Query"

@app.route('/newpage', methods=['GET', 'POST'])
def newpage():
    if request.method == 'POST':
        user_input = request.form['user_input']
        svm_result = detect_sql_injection(user_input, svm_classifier)
        return render_template('result.html', result=svm_result)
    return render_template('newpage.html')

@socketio.on('connect')
def handle_connect():
    """Handles new client connection"""
    try:
        client_ip = request.remote_addr
        location = get_geo_location(client_ip)
        connected_users[request.sid] = {
            'ip': client_ip,
            'location': location['location'],
            'lat': location['lat'],
            'lon': location['lon']
        }
        logging.info(f"New client connected: {client_ip} from {location['location']}")
        emit('update_users', connected_users, broadcast=True)
    except Exception as e:
        logging.error(f"Error during connection: {str(e)}")

@socketio.on('disconnect')
def handle_disconnect():
    """Handles client disconnection"""
    try:
        if request.sid in connected_users:
            user = connected_users.pop(request.sid)
            logging.info(f"Client disconnected: {user['ip']} from {user['location']}")
            emit('update_users', connected_users, broadcast=True)
    except Exception as e:
        logging.error(f"Error during disconnect: {str(e)}")

if __name__ == '__main__':
    eventlet.wsgi.server(eventlet.listen(("0.0.0.0", 5000)), app)

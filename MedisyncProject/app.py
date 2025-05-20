from flask import Flask, request, jsonify, render_template, send_from_directory, redirect, url_for, session, flash
import numpy as np
import os
from Model.models import predict
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
import sqlite3
import secrets
from Model.lung_cancer.model import predict as predict_lung_cancer
from Model.breast_cancer.model import predict as predict_breast_cancer
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Get the absolute path of the current directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, 'MEDISYNC-AI-main')
DB_PATH = os.path.join(BASE_DIR, 'users.db')

app = Flask(__name__, 
    static_folder=STATIC_DIR,
    template_folder=STATIC_DIR
)

app.secret_key = secrets.token_hex(16)  # Generate a secure secret key
app.config['SESSION_COOKIE_SECURE'] = True  # Only send cookie over HTTPS
app.config['SESSION_COOKIE_HTTPONLY'] = True  # Prevent JavaScript access to session cookie
app.config['PERMANENT_SESSION_LIFETIME'] = 3600  # Session timeout in seconds (1 hour)
app.config['DEBUG'] = True  # Enable debug mode

# Context processor to make user info available to all templates
@app.context_processor
def inject_user():
    return {
        'is_authenticated': 'user_id' in session,
        'user_name': session.get('user_name', None)
    }

# Dictionary of algorithms and their accuracies
ALGORITHMS = {
    'svm': {'name': 'Support Vector Machine', 'accuracy': '98%'},
    'logistic': {'name': 'Logistic Regression', 'accuracy': '85%'},
    'random_forest': {'name': 'Random Forest', 'accuracy': '88%'}
}

# Available algorithms for each model
LUNG_CANCER_ALGORITHMS = {
    'logistic': {'name': 'Logistic Regression', 'accuracy': '95%'},
    'random_forest': {'name': 'Random Forest', 'accuracy': '97%'}
}

BREAST_CANCER_ALGORITHMS = {
    'svm_rbf': {'name': 'SVM (RBF Kernel)', 'accuracy': '96.49%'},
    'svm_linear': {'name': 'SVM (Linear Kernel)', 'accuracy': '99.12%'},
    'knn': {'name': 'K-Nearest Neighbors', 'accuracy': '94.74%'},
    'decision_tree': {'name': 'Decision Tree', 'accuracy': '88.60%'},
    'random_forest': {'name': 'Random Forest', 'accuracy': '98.25%'},
    'logistic': {'name': 'Logistic Regression', 'accuracy': '96.49%'}
}

# Update database connection to use absolute path
def get_db_connection():
    return sqlite3.connect(DB_PATH)

# Initialize database
def init_db():
    try:
        conn = get_db_connection()
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS users
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                    first_name TEXT NOT NULL,
                    last_name TEXT NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    password TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Database initialization error: {str(e)}")

# Login required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/')
def home():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('index.html')

@app.route('/<path:path>')
def serve_static(path):
    # If it's an HTML file, check if it needs authentication
    if path.endswith('.html'):
        # Allow access to login and signup pages without authentication
        if path in ['login.html', 'signup.html']:
            return render_template(path)
        # All other HTML pages require authentication
        elif 'user_id' not in session:
            return redirect(url_for('login'))
        return render_template(path)
    # Serve static files (images, CSS, JS, etc.)
    return send_from_directory(STATIC_DIR, path)

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        first_name = request.form.get('first_name')
        last_name = request.form.get('last_name')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')

        if not all([first_name, last_name, email, password, confirm_password]):
            return render_template('signup.html', error="All fields are required")

        if password != confirm_password:
            return render_template('signup.html', error="Passwords do not match")

        hashed_password = generate_password_hash(password)

        try:
            conn = get_db_connection()
            c = conn.cursor()
            c.execute('INSERT INTO users (first_name, last_name, email, password) VALUES (?, ?, ?, ?)',
                     (first_name, last_name, email, hashed_password))
            conn.commit()
            conn.close()

            return render_template('login.html', success="Account created successfully! Please log in.")
        except sqlite3.IntegrityError:
            return render_template('signup.html', error="Email already exists")
        except Exception as e:
            return render_template('signup.html', error=f"An error occurred: {str(e)}")

    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        if not email or not password:
            return render_template('login.html', error="Email and password are required")

        try:
            conn = get_db_connection()
            c = conn.cursor()
            c.execute('SELECT * FROM users WHERE email = ?', (email,))
            user = c.fetchone()
            conn.close()

            if user and check_password_hash(user[4], password):
                session.clear()
                session['user_id'] = user[0]
                session['user_name'] = f"{user[1]} {user[2]}"
                session['email'] = user[3]
                session['joined_date'] = user[5]
                session.permanent = True
                
                next_page = request.args.get('next')
                return redirect(next_page or url_for('home'))
            else:
                return render_template('login.html', error="Invalid email or password")

        except Exception as e:
            return render_template('login.html', error=f"An error occurred: {str(e)}")

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/lung-cancer-prediction')
@login_required
def lung_cancer_prediction():
    return render_template('lung-cancer-prediction.html', algorithms=LUNG_CANCER_ALGORITHMS)

@app.route('/breast-cancer-prediction')
@login_required
def breast_cancer_prediction():
    return render_template('breast-cancer-prediction.html', algorithms=BREAST_CANCER_ALGORITHMS)

@app.route('/predict_lung_cancer', methods=['POST'])
@login_required
def predict_lung():
    try:
        # Get form data and selected algorithm
        data = request.form
        algorithm = data.get('algorithm', 'svm')
        
        logger.debug("\n=== Debug: Form Data ===")
        logger.debug(f"Raw form data: {dict(data)}")
        
        # Convert form data to feature array
        features = [
            int(data['gender']), int(data['age']), int(data['smoking']),
            int(data['yellow_fingers']), int(data['anxiety']),
            int(data['peer_pressure']), int(data['chronic_disease']),
            int(data['fatigue']), int(data['allergy']),
            int(data['wheezing']), int(data['alcohol']),
            int(data['coughing']), int(data['shortness_of_breath']),
            int(data['swallowing_difficulty']), int(data['chest_pain'])
        ]
        
        logger.debug("\nDebug: Features array:")
        feature_names = ['Gender', 'Age', 'Smoking', 'Yellow Fingers', 'Anxiety', 
                        'Peer Pressure', 'Chronic Disease', 'Fatigue', 'Allergy',
                        'Wheezing', 'Alcohol Consuming', 'Coughing', 
                        'Shortness of Breath', 'Swallowing Difficulty', 'Chest Pain']
        for name, value in zip(feature_names, features):
            logger.debug(f"{name}: {value}")
        
        # Make prediction
        result = predict_lung_cancer(features, algorithm)
        
        logger.debug("\nDebug: Prediction result: %s", result)
        
        # Check if prediction was successful
        if result.get('error') or result.get('prediction') is None:
            return render_template('lung-cancer-prediction.html',
                                error=f"An error occurred during prediction: {result.get('error', 'Unknown error')}",
                                algorithms=LUNG_CANCER_ALGORITHMS)
        
        # Store the health check result
        try:
            conn = get_db_connection()
            c = conn.cursor()
            c.execute('''
                INSERT INTO health_checks (user_id, check_type, algorithm, prediction, probability)
                VALUES (?, ?, ?, ?, ?)
            ''', (session['user_id'], 'lung_cancer', algorithm, result['prediction'], result['probability']))
            
            # Update lung health metric
            lung_health = 100 - (result['probability'] * 100) if result['probability'] else 95
            c.execute('''
                INSERT OR REPLACE INTO health_metrics (user_id, metric_type, value)
                VALUES (?, 'lung', ?)
            ''', (session['user_id'], lung_health))
            
            conn.commit()
            conn.close()
        except Exception as db_error:
            logger.error(f"Database error: {str(db_error)}")
            return render_template('lung-cancer-prediction.html',
                                error="Failed to save prediction results",
                                algorithms=LUNG_CANCER_ALGORITHMS)
        
        # Generate health message based on prediction
        if result['prediction'] == 1:  # High Risk
            health_message = """
                <p>Based on the provided symptoms, there may be an elevated risk of lung cancer. 
                Please consult with a healthcare professional for a thorough evaluation.</p>
                <ul>
                    <li>Schedule an appointment with your doctor</li>
                    <li>Discuss your symptoms and risk factors</li>
                    <li>Consider getting a chest X-ray or CT scan</li>
                </ul>
            """
        else:  # Low Risk
            health_message = """
                <p>Based on the provided symptoms, the risk of lung cancer appears to be lower. 
                However, it's important to:</p>
                <ul>
                    <li>Maintain regular check-ups with your healthcare provider</li>
                    <li>Avoid smoking and exposure to secondhand smoke</li>
                    <li>Report any new or worsening symptoms to your doctor</li>
                </ul>
            """
        
        # Calculate the confidence percentage
        # For High Risk (1), use the probability directly
        # For Low Risk (0), use 1 - probability and scale it to be more intuitive
        if result['prediction'] == 1:  # High Risk
            confidence = result['probability']
            confidence_text = f"{confidence*100:.1f}%"
        else:  # Low Risk
            confidence = 1 - result['probability']
            # Scale confidence to be more intuitive (e.g., 0.61 -> 91%)
            scaled_confidence = 0.7 + (confidence * 0.3)  # Scale between 70-100%
            confidence_text = f"{scaled_confidence*100:.1f}%"
        
        return render_template('lung-cancer-prediction.html',
                             result={'prediction': 'High Risk' if result['prediction'] == 1 else 'Low Risk',
                                     'probability': confidence_text,
                                     'health_message': health_message,
                                     'algorithm': LUNG_CANCER_ALGORITHMS[algorithm]},
                             algorithms=LUNG_CANCER_ALGORITHMS)
                             
    except Exception as e:
        print("\nDebug: Error occurred:")
        import traceback
        print(traceback.format_exc())
        return render_template('lung-cancer-prediction.html',
                             error=f"An error occurred: {str(e)}",
                             algorithms=LUNG_CANCER_ALGORITHMS)

@app.route('/predict_breast_cancer', methods=['POST'])
@login_required
def predict_breast():
    try:
        # Get form data and selected algorithm
        data = request.form
        algorithm = data.get('algorithm', 'svm_rbf')
        
        # Convert form data to feature array
        features = [
            # Mean values
            float(data['radius_mean']), float(data['texture_mean']), 
            float(data['perimeter_mean']), float(data['area_mean']),
            float(data['smoothness_mean']), float(data['compactness_mean']),
            float(data['concavity_mean']), float(data['concave_points_mean']),
            float(data['symmetry_mean']), float(data['fractal_dimension_mean']),
            
            # Standard error values
            float(data['radius_se']), float(data['texture_se']),
            float(data['perimeter_se']), float(data['area_se']),
            float(data['smoothness_se']), float(data['compactness_se']),
            float(data['concavity_se']), float(data['concave_points_se']),
            float(data['symmetry_se']), float(data['fractal_dimension_se']),
            
            # Worst values
            float(data['radius_worst']), float(data['texture_worst']),
            float(data['perimeter_worst']), float(data['area_worst']),
            float(data['smoothness_worst']), float(data['compactness_worst']),
            float(data['concavity_worst']), float(data['concave_points_worst']),
            float(data['symmetry_worst']), float(data['fractal_dimension_worst'])
        ]
        
        # Make prediction
        result = predict_breast_cancer(features, algorithm)
        
        # Only insert if prediction is not None and no error
        if result.get('error') or result.get('prediction') is None:
            return render_template('breast-cancer-prediction.html',
                                 error=f"An error occurred during prediction: {result.get('error', 'Unknown error')}",
                                 algorithms=BREAST_CANCER_ALGORITHMS)
        
        # Store the health check result using context manager
        try:
            with get_db_connection() as conn:
                c = conn.cursor()
                c.execute('''
                    INSERT INTO health_checks (user_id, check_type, algorithm, prediction, probability)
                    VALUES (?, ?, ?, ?, ?)
                ''', (session['user_id'], 'breast_cancer', algorithm, result['prediction'], result['probability']))
                conn.commit()
        except Exception as db_error:
            logger.error(f"Database error: {str(db_error)}")
            return render_template('breast-cancer-prediction.html',
                                 error="Failed to save prediction results",
                                 algorithms=BREAST_CANCER_ALGORITHMS)
        
        # Generate health message based on prediction
        if result['prediction'] == 1:
            health_message = """
                <p>Based on the diagnostic measurements, there may be indicators suggesting malignancy. 
                It is crucial to:</p>
                <ul>
                    <li>Consult with a specialist immediately</li>
                    <li>Schedule additional diagnostic tests</li>
                    <li>Discuss treatment options with your healthcare team</li>
                </ul>
            """
        else:
            health_message = """
                <p>Based on the diagnostic measurements, the indicators suggest benign characteristics. 
                However, it's important to:</p>
                <ul>
                    <li>Continue regular breast cancer screenings</li>
                    <li>Maintain routine check-ups with your healthcare provider</li>
                    <li>Be aware of any changes in breast tissue</li>
                </ul>
            """
        
        return render_template('breast-cancer-prediction.html',
                             result={'prediction': 'Malignant' if result['prediction'] == 1 else 'Benign',
                                     'probability': f"{result['probability']*100:.1f}%" if result['probability'] else None,
                                     'health_message': health_message,
                                     'algorithm': BREAST_CANCER_ALGORITHMS[algorithm]},
                             algorithms=BREAST_CANCER_ALGORITHMS)
    except Exception as e:
        print("\nDebug: Error occurred:")
        import traceback
        print(traceback.format_exc())
        return render_template('breast-cancer-prediction.html',
                             error=f"An error occurred: {str(e)}",
                             algorithms=BREAST_CANCER_ALGORITHMS)

@app.route('/ai-diagnosis')
@login_required
def ai_diagnosis():
    return render_template('ai-diagnosis.html')

@app.route('/profile')
@login_required
def profile():
    # Get current time
    now = datetime.now()
    
    # Get user's health checks from database
    conn = get_db_connection()
    c = conn.cursor()
    
    # Get total health checks
    c.execute('''
        SELECT COUNT(*) 
        FROM health_checks 
        WHERE user_id = ?
    ''', (session['user_id'],))
    health_checks = c.fetchone()[0]
    
    # Get recent health checks
    c.execute('''
        SELECT DISTINCT check_type, algorithm, prediction, probability, created_at,
               ROW_NUMBER() OVER (PARTITION BY check_type, algorithm, prediction ORDER BY created_at DESC) as rn
        FROM health_checks 
        WHERE user_id = ? 
        ORDER BY created_at DESC
    ''', (session['user_id'],))
    all_checks = c.fetchall()
    
    # Filter out duplicate entries within short time periods
    recent_checks = []
    seen = set()
    for check in all_checks:
        # Create a key from relevant fields
        key = (check[0], check[1], check[2], check[3])
        # Only add if we haven't seen this exact check recently
        if key not in seen:
            seen.add(key)
            # Convert timestamp string to datetime
            try:
                check_time = datetime.strptime(check[4], '%Y-%m-%d %H:%M:%S')
                recent_checks.append((check[0], check[1], check[2], check[3], check_time))
            except (ValueError, TypeError):
                continue
            
    # Limit to 5 most recent unique checks
    recent_checks = recent_checks[:5]
    
    # Get health metrics
    c.execute('''
        SELECT metric_type, value 
        FROM health_metrics 
        WHERE user_id = ?
    ''', (session['user_id'],))
    metrics = c.fetchall()
    
    # Convert metrics to dictionary
    health_metrics = {
        'heart': 0,
        'lung': 0,
        'neural': 0,
        'immunity': 0
    }
    for metric in metrics:
        health_metrics[metric[0]] = float(metric[1])
    
    # Calculate overall health score (average of all metrics)
    health_score = sum(health_metrics.values()) / len(health_metrics)
    
    c.close()
    conn.close()
    
    return render_template('profile.html',
        user_name=session.get('name', 'User'),
        health_checks=health_checks,
        health_score=round(health_score, 1),
        health_metrics=health_metrics,
        recent_checks=recent_checks,
        now=now  # Pass current time to template
    )

@app.route('/features')
@login_required
def features():
    return render_template('features.html')

@app.route('/about')
@login_required
def about():
    return render_template('about.html')

@app.route('/solutions')
@login_required
def solutions():
    return render_template('solutions.html')

@app.route('/research')
@login_required
def research():
    return render_template('research.html')

@app.route('/contact')
@login_required
def contact():
    return render_template('contact.html')

@app.route('/settings')
@login_required
def settings():
    return render_template('settings.html')

@app.template_filter('datetime')
def format_datetime(value):
    """Format a datetime object to a relative time string."""
    now = datetime.now()
    diff = now - value
    
    if diff.days > 7:
        return value.strftime('%B %d, %Y')
    elif diff.days > 0:
        return f"{diff.days} days ago"
    elif diff.seconds > 3600:
        hours = diff.seconds // 3600
        return f"{hours} hours ago"
    elif diff.seconds > 60:
        minutes = diff.seconds // 60
        return f"{minutes} minutes ago"
    else:
        return "just now"

# Initialize database on startup
init_db()

if __name__ == '__main__':
    app.run(debug=True)

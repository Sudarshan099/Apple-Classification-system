from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_bcrypt import Bcrypt
from flask_mysqldb import MySQL
import tensorflow as tf
import joblib
import numpy as np

from werkzeug.utils import secure_filename
import os
from tensorflow.keras.applications import DenseNet201
from sklearn.metrics import accuracy_score

# Flask app setup
app = Flask(__name__)
app.secret_key = 'session_for_apple'

# Database configuration
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'user_db'
app.config['MYSQL_PORT'] = 3306

mysql = MySQL(app)
bcrypt = Bcrypt(app)

# Load models
try:
    densenet_model = tf.keras.models.load_model('models/densenet201_transfer_learning_model.keras')
    knn_model = joblib.load('models/knn_model.joblib')
except Exception as e:
    print(f"Error loading models: {e}")
    raise

# Allowed file extensions
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Helper function to validate uploaded file
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Routes
@app.route('/')
def home(): # Debug login state
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    """Handle user registration."""
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')

        try:
            cur = mysql.connection.cursor()
            cur.execute("INSERT INTO users (username, password) VALUES (%s, %s)", (username, hashed_password))
            mysql.connection.commit()
            cur.close()

            flash('Registration successful. Please log in.', 'success')
            return redirect(url_for('home'))
        except Exception as e:
            flash(f"Error during registration: {e}", 'danger')
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Handle user login."""
    if request.method == 'POST':
        username = request.form.get('username').strip()  # Strip whitespace
        password = request.form.get('password')

        if not username or not password:
            flash('Username and password are required.', 'warning')
            return redirect(url_for('login'))

        try:
            # Connect to the database and fetch user details
            cur = mysql.connection.cursor()
            cur.execute("SELECT username, password FROM users WHERE username = %s", (username,))
            user = cur.fetchone()
            cur.close()

            # Validate credentials
            if user and bcrypt.check_password_hash(user[1], password):  # Assuming password hash is in the second column
                session['username'] = username  # Store username in the session
                print("Session set after login:", session) 
                flash('Login successful', 'success')
                return redirect(url_for('upload'))  # Redirect to the upload page
            else:
                flash('Invalid username or password.', 'danger')

        except Exception as e:
            flash(f"An error occurred during login: {e}", 'danger')
            app.logger.error(f"Login error: {e}")  # Log the error for debugging

    return render_template('login.html')



@app.route('/upload', methods=['GET', 'POST'])
def upload():
    """Handle file upload and prediction."""
    if 'username' not in session:
        flash('Please log in to access this page.', 'warning')
        return redirect(url_for('home'))

    predicted_class = None
    confidence_score = None

    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file uploaded.', 'danger')
            return redirect(request.url)

        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            try:
                # Preprocess the image
                image = tf.keras.preprocessing.image.load_img(filepath, target_size=(224, 224))
                image_array = tf.keras.preprocessing.image.img_to_array(image) / 255.0
                image_array = np.expand_dims(image_array, axis=0)

                # Extract features
                feature_extractor = DenseNet201(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
                feature_extractor.trainable = False
                features = feature_extractor.predict(image_array)
                features = features.reshape(1, -1)

                # Predict class using KNN
                prediction_probabilities = knn_model.predict_proba(features)
                confidence_score = np.max(prediction_probabilities) * 100  # Convert to percentage

                if confidence_score >= 85:
                    predicted_class_index = np.argmax(prediction_probabilities)
                    idx_to_class = {0: 'Apple Braeburn',
                                    1: 'Apple Crimson Snow',
                                    2: 'Apple Golden 1',
                                    3: 'Apple Golden 2',
                                    4: 'Apple Golden 3',
                                    5: 'Apple Granny Smith',
                                    6: 'Apple Pink Lady',
                                    7: 'Apple Red 1',
                                    8: 'Apple Red 2',
                                    9: 'Apple Red 3',
                                    10: 'Apple Red Delicious',
                                    11: 'Apple Red Yellow 1',
                                    12: 'Apple Red Yellow 2'}
                    predicted_class = idx_to_class.get(predicted_class_index, "Unknown")
                    flash(f'Predicted class: {predicted_class} with confidence: {confidence_score:.2f}%', 'success')
                else:
                    predicted_class = "Please choose a proper apple image"
                    flash('The uploaded image is not recognized as an apple with sufficient confidence.', 'warning')
            except Exception as e:
                flash(f'Error during prediction: {str(e)}', 'danger')
            finally:
                os.remove(filepath)  # Remove the uploaded file after processing
        else:
            flash('Invalid file type. Allowed types: jpg, jpeg, png.', 'danger')

    return render_template(
        'upload.html',
        predicted_class=predicted_class,
        confidence_score=confidence_score
    )

@app.route('/logout')
def logout():
    """Log out the user and clear the session."""
    session.pop('username', None)
    flash('You have been logged out.', 'info')
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=False)

from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_bcrypt import Bcrypt
from flask_mysqldb import MySQL
import tensorflow as tf
import joblib
import numpy as np
from werkzeug.utils import secure_filename
import os
from tensorflow.keras.applications import DenseNet201

# Flask app setup
app = Flask(__name__)
app.secret_key = 'secure_key_for_session'

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
def home():
    """Render the login page."""
    return render_template('login.html')

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
        username = request.form['username']
        password = request.form['password']

        try:
            cur = mysql.connection.cursor()
            cur.execute("SELECT * FROM users WHERE username = %s", [username])
            user = cur.fetchone()
            cur.close()

            if user and bcrypt.check_password_hash(user[2], password):
                session['username'] = username
                flash('Login successful', 'success')
                return redirect(url_for('upload'))
            else:
                flash('Invalid credentials', 'danger')
        except Exception as e:
            flash(f"Error during login: {e}", 'danger')
    return render_template('login.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    """Handle file upload and prediction."""
    if 'username' not in session:
        flash('Please log in to access this page.', 'warning')
        return redirect(url_for('home'))

    predicted_class = None

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
                feature_extractor.trainable = False  # Freeze DenseNet201
                features = feature_extractor.predict(image_array)
                features = features.reshape(1, -1)

                # Predict class
                prediction = knn_model.predict(features)
                predicted_class = list(knn_model.classes_)[prediction[0]]
                idx_to_class = {0: 'apple_6',
                    1: 'apple_braeburn_1',
                    2: 'apple_crimson_snow_1',
                    3: 'apple_golden_1',
                    4: 'apple_golden_2',
                    5: 'apple_golden_3',
                    6: 'apple_granny_smith_1',
                    7: 'apple_hit_1',
                    8: 'apple_pink_lady_1',
                    9: 'apple_red_1',
                    10: 'apple_red_2',
                    11: 'apple_red_3',
                    12: 'apple_red_delicios_1',
                    13: 'apple_red_yellow_1',
                    14: 'apple_rotten_1'}
                flash(f'Predicted class: {idx_to_class[predicted_class]}', 'success')
            except Exception as e:
                flash(f'Error during prediction: {str(e)}', 'danger')
            finally:
                os.remove(filepath)  # Remove the uploaded file after processing
        else:
            flash('Invalid file type. Allowed types: jpg, jpeg, png.', 'danger')

    return render_template('upload.html', predicted_class=predicted_class)

@app.route('/logout')
def logout():
    """Log out the user and clear the session."""
    session.pop('username', None)
    flash('You have been logged out.', 'info')
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)

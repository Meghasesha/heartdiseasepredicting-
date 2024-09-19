from flask import Flask, request, render_template, redirect, url_for, flash
from flask_login import (
    LoginManager,
    UserMixin,
    login_user,
    login_required,
    logout_user,
    current_user,
)
from werkzeug.security import generate_password_hash, check_password_hash
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import sessionmaker, declarative_base
import os
import logging
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'your_default_secret_key')

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# Flask-Login setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Database setup using SQLite with SQLAlchemy
DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///users.db')

# Create SQLAlchemy engine and session
engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

# User model
class User(UserMixin, Base):
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True)
    username = Column(String(150), unique=True, nullable=False)
    password = Column(String(255), nullable=False)

    def get_id(self):
        return str(self.id)

# Create the database tables
Base.metadata.create_all(engine)

@login_manager.user_loader
def load_user(user_id):
    session = SessionLocal()
    user = session.query(User).get(int(user_id))
    session.close()
    return user

# Load the heart disease data from CSV and train the model
CSV_PATH = 'heart_disease_data.csv'

def load_and_train_model():
    try:
        logging.info(f"Loading data from {CSV_PATH}")
        data = pd.read_csv(CSV_PATH)
        
        # Prepare features and labels
        X = data.drop('output', axis=1)
        y = data['output']
        
        # Split the dataset into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train the model (RandomForest in this case)
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        
        logging.info("Model trained successfully")
        return model
    except FileNotFoundError as e:
        logging.error(f"CSV file not found: {e}")
        raise e
    except Exception as e:
        logging.error(f"Error during model training: {e}")
        raise e

# Train the model at the start of the app
model = None
try:
    model = load_and_train_model()
except Exception as e:
    logging.error(f"Error loading and training the model: {e}")

# Registration Route
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        # Ensure passwords match
        if password != confirm_password:
            flash('Passwords do not match!', 'danger')
            return render_template('register.html')

        # Hash the password
        hashed_password = generate_password_hash(password)

        session = SessionLocal()
        try:
            # Check if username already exists
            existing_user = session.query(User).filter_by(username=username).first()
            if existing_user:
                flash('Username already exists!', 'danger')
                return render_template('register.html')

            # Create new user
            new_user = User(username=username, password=hashed_password)
            session.add(new_user)
            session.commit()

            flash('Registration successful! Please log in.', 'success')
            return redirect(url_for('login'))

        except Exception as e:
            session.rollback()
            logging.error(f"Error during registration: {e}")
            flash('An error occurred during registration. Please try again.', 'danger')
            return render_template('register.html')
        finally:
            session.close()
    return render_template('register.html')

# Login Route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        session = SessionLocal()
        try:
            user = session.query(User).filter_by(username=username).first()
            if user and check_password_hash(user.password, password):
                login_user(user)
                flash('Login successful!', 'success')
                return redirect(url_for('home'))
            else:
                flash('Invalid username or password', 'danger')
                return render_template('login.html')
        except Exception as e:
            logging.error(f"Error during login: {e}")
            flash('An error occurred during login. Please try again.', 'danger')
            return render_template('login.html')
        finally:
            session.close()
    return render_template('login.html')

# Logout Route
@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

# Home Route
@app.route('/')
@login_required
def home():
    return render_template('index.html')

# Prediction Route
@app.route('/predict', methods=['POST'])
@login_required
def predict():
    try:
        logging.info(f"Processing prediction for user: {current_user.username}")

        # Retrieve form data
        form_values = [
            request.form.get(key, type=float)
            for key in [
                'age', 'sex', 'cp', 'trtbps', 'chol', 'fbs', 'restecg',
                'thalachh', 'exng', 'oldpeak', 'slp', 'caa', 'thall'
            ]
        ]

        features = np.array([form_values])

        # Make prediction
        prediction = model.predict(features)

        output = "No Heart Disease" if prediction[0] == 0 else "Heart Disease"
        logging.info(f"Prediction result: {output} for user {current_user.username}")
        return render_template('index.html', prediction_text=f'Prediction: {output}')
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        flash('An error occurred during prediction. Please try again.', 'danger')
        return redirect(url_for('home'))

if __name__ == "__main__":
    logging.info("Starting Flask app...")
    app.run(debug=True)

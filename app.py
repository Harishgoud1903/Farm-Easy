from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user
import joblib
import numpy as np
import os

# -------------------- Initialize Flask --------------------
app = Flask(__name__)
app.secret_key = 'your_secret_key'

# -------------------- Database Setup --------------------
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
db = SQLAlchemy(app)

# -------------------- Flask-Login Setup --------------------
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# -------------------- User Model --------------------
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), nullable=False, unique=True)
    password = db.Column(db.String(150), nullable=False)

with app.app_context():
    db.create_all()

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))  # Updated for SQLAlchemy 2.0

# -------------------- Load Model and Encoder --------------------
try:
    model_path = os.path.join('model', 'crop_model.pkl')
    encoder_path = os.path.join('model', 'encoder.pkl')

    model = joblib.load(model_path)
    encoder = joblib.load(encoder_path)

    print("✅ Model and encoder loaded successfully!")
    print("➡️ Encoder classes:", getattr(encoder, "classes_", None))
except Exception as e:
    print("❌ Error loading model or encoder:", e)

# -------------------- Home / Landing Page --------------------
@app.route('/')
def home():
    return render_template('home.html')

# -------------------- Register --------------------
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if User.query.filter_by(username=username).first():
            flash('Username already exists!', 'danger')
            return redirect(url_for('register'))

        new_user = User(username=username, password=password)
        db.session.add(new_user)
        db.session.commit()
        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('login'))

    return render_template('register.html')

# -------------------- Login --------------------
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        user = User.query.filter_by(username=username, password=password).first()
        if user:
            login_user(user)
            return redirect(url_for('predict'))
        else:
            flash('Invalid username or password', 'danger')
            return redirect(url_for('login'))

    return render_template('login.html')

# -------------------- Logout --------------------
@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Logged out successfully.', 'success')
    return redirect(url_for('login'))

# -------------------- Crop Prediction --------------------
@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    prediction = None
    if request.method == 'POST':
        try:
            # Collect user inputs
            N = float(request.form['N'])
            P = float(request.form['P'])
            K = float(request.form['K'])
            temperature = float(request.form['temperature'])
            humidity = float(request.form['humidity'])
            ph = float(request.form['ph'])
            rainfall = float(request.form['rainfall'])

            # Prepare features
            features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
            print("➡️ Input features:", features)

            # Predict
            encoded_prediction = model.predict(features)
            print("➡️ Encoded prediction:", encoded_prediction)

            # Handle model output (numeric or string)
            if isinstance(encoded_prediction[0], str):
                predicted_crop = encoded_prediction[0]
            else:
                predicted_crop = encoder.inverse_transform(encoded_prediction)[0]

            print("✅ Predicted crop:", predicted_crop)

            prediction = predicted_crop

        except Exception as e:
            print("❌ Error during prediction:", e)
            flash(f"Error during prediction: {e}", "danger")

    return render_template('predict.html', prediction=prediction)

# -------------------- Run Flask App --------------------
if __name__ == '__main__':
    app.run(debug=True)

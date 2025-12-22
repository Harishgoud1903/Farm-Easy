import os
import re
import joblib
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import (
    LoginManager, UserMixin,
    login_user, logout_user,
    login_required
)
from werkzeug.security import generate_password_hash, check_password_hash
from urllib.parse import urlparse, urljoin

# ------------------------
# App setup
# ------------------------
app = Flask(__name__)
app.secret_key = "replace_with_a_real_secret"

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DB_PATH = os.path.join(BASE_DIR, "users.db")

app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///" + DB_PATH.replace("\\", "/")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

login_manager = LoginManager(app)
login_manager.login_view = "login"

# ------------------------
# Database Model
# ------------------------
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(300), nullable=False)

with app.app_context():
    db.create_all()

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

# ------------------------
# Password Validation
# ------------------------
def is_valid_password(username, password):
    if username == password:
        return False, "Password cannot be same as username"

    pattern = r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&]).{8,}$'
    if not re.match(pattern, password):
        return False, (
            "Password must be at least 8 characters and include "
            "uppercase, lowercase, number and special character"
        )

    return True, ""

# ------------------------
# Safe redirect
# ------------------------
def is_safe_url(target):
    host_url = urlparse(request.host_url)
    redirect_url = urlparse(urljoin(request.host_url, target))
    return redirect_url.scheme in ("http", "https") and host_url.netloc == redirect_url.netloc

# ------------------------
# Load ML Model
# ------------------------
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_FILE = os.path.join(MODELS_DIR, "crop_model.pkl")
ENCODER_FILE = os.path.join(MODELS_DIR, "encoder.pkl")

model = None
encoder = None

try:
    model = joblib.load(MODEL_FILE)
    print("✔ Crop model loaded")
except Exception as e:
    print("⚠ Model not loaded:", e)

try:
    encoder = joblib.load(ENCODER_FILE)
    print("✔ Encoder loaded")
except Exception:
    encoder = None

def safe_inverse_transform(pred):
    try:
        if encoder:
            return encoder.inverse_transform(pred)[0]
    except Exception:
        pass
    return str(pred[0])

def image_exists(filename):
    return os.path.exists(os.path.join(BASE_DIR, "static", "images", filename or ""))

# ------------------------
# CROPS DICTIONARY (FIXED POSITION)
# ------------------------
CROPS = {
    "apple": {"img":"apple.jpg","description":"Apple requires cool climates and well-drained soil."},
    "banana": {"img":"banana.jpg","description":"Banana grows well in humid tropical climates."},
    "blackgram": {"img":"blackgram.jpg","description":"Blackgram is an important pulse crop."},
    "chickpea": {"img":"chickpea.jpg","description":"Chickpea grows best in dry climates."},
    "coconut": {"img":"coconut.jpg","description":"Coconut thrives in coastal regions."},
    "coffee": {"img":"coffee.jpg","description":"Coffee grows in cool shaded areas."},
    "cotton": {"img":"cotton.jpg","description":"Cotton is a major fiber crop."},
    "grapes": {"img":"grapes.jpg","description":"Grapes grow well in dry warm climates."},
    "jute": {"img":"jute.jpg","description":"Jute is used for fiber production."},
    "maize": {"img":"maize.jpg","description":"Maize grows in warm climates."},
    "mango": {"img":"mango.jpg","description":"Mango is a tropical fruit crop."},
    "rice": {"img":"rice.jpg","description":"Rice grows in water-abundant regions."},
    "watermelon": {"img":"watermelon.jpg","description":"Watermelon grows in hot climates."}
}

# ------------------------
# Routes
# ------------------------
@app.route("/")
def home():
    return render_template("home.html")

# ------------------------
# Register
# ------------------------
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        if not username or not password:
            flash("Username and password required", "warning")
            return redirect(url_for("register"))

        if User.query.filter_by(username=username).first():
            flash("Username already exists", "danger")
            return redirect(url_for("login"))

        valid, msg = is_valid_password(username, password)
        if not valid:
            flash(msg, "danger")
            return redirect(url_for("register"))

        hashed_password = generate_password_hash(password)
        user = User(username=username, password=hashed_password)

        db.session.add(user)
        db.session.commit()

        flash("Registration successful. Please login.", "success")
        return redirect(url_for("login"))

    return render_template("register.html")

# ------------------------
# Login
# ------------------------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        next_page = request.form.get("next") or request.args.get("next")

        user = User.query.filter_by(username=username).first()
        if not user or not check_password_hash(user.password, password):
            flash("Invalid username or password", "danger")
            return redirect(url_for("login"))

        login_user(user)
        flash("Login successful", "success")

        if next_page and is_safe_url(next_page):
            return redirect(next_page)

        return redirect(url_for("crops"))

    return render_template("login.html")

# ------------------------
# Logout
# ------------------------
@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash("Logged out successfully", "info")
    return redirect(url_for("home"))

# ------------------------
# Crops Page (ERROR FIXED)
# ------------------------
@app.route("/crops")
@login_required
def crops():
    crops_with_images = {}
    for k, v in CROPS.items():
        data = v.copy()
        if not image_exists(data.get("img", "")):
            data["img"] = "crop.jpg"
        crops_with_images[k] = data

    return render_template("crops.html", crops=crops_with_images)

# ------------------------
# Prediction
# ------------------------
@app.route("/predict", methods=["GET", "POST"])
@login_required
def predict():
    if request.method == "POST":
        try:
            N = float(request.form.get("N"))
            P = float(request.form.get("P"))
            K = float(request.form.get("K"))
            temperature = float(request.form.get("temperature"))
            humidity = float(request.form.get("humidity"))
            ph = float(request.form.get("ph"))
            rainfall = float(request.form.get("rainfall"))
        except Exception:
            flash("Enter valid numeric values", "danger")
            return redirect(url_for("predict"))

        features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        prediction = model.predict(features)
        crop = safe_inverse_transform(prediction)

        return render_template("predict.html", prediction=crop)

    return render_template("predict.html", prediction=None)

# ------------------------
# Run
# ------------------------
if __name__ == "__main__":
    app.run(debug=True)

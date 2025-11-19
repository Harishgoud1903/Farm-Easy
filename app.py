import os
import joblib
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from urllib.parse import urlparse, urljoin

# ------------- App setup -------------
app = Flask(__name__)
app.secret_key = "replace_with_a_real_secret"

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DB_PATH = os.path.join(BASE_DIR, "users.db")
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///" + DB_PATH.replace("\\", "/")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = "login"

# ------------- Models -------------
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(300), nullable=False)  # hashed

with app.app_context():
    db.create_all()

@login_manager.user_loader
def load_user(user_id):
    try:
        return db.session.get(User, int(user_id))
    except Exception:
        return None

# ------------- ML model load (optional) -------------
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_FILE = os.path.join(MODELS_DIR, "crop_model.pkl")
ENCODER_FILE = os.path.join(MODELS_DIR, "encoder.pkl")
model = None
encoder = None
try:
    model = joblib.load(MODEL_FILE)
    print("✔ Model loaded")
except Exception as e:
    print("⚠ Could not load model:", e)
try:
    encoder = joblib.load(ENCODER_FILE)
    print("✔ Encoder loaded")
except Exception:
    encoder = None

# ------------- Helpers -------------
def is_safe_url(target):
    host_url = urlparse(request.host_url)
    redirect_url = urlparse(urljoin(request.host_url, target))
    return (redirect_url.scheme in ("http","https")) and (host_url.netloc == redirect_url.netloc)

def safe_inverse_transform(pred):
    try:
        if encoder is not None:
            return encoder.inverse_transform(pred)[0]
    except Exception:
        pass
    try:
        return str(pred[0])
    except Exception:
        return str(pred)

def image_exists(filename):
    return os.path.exists(os.path.join(BASE_DIR, "static", "images", filename or ""))

# ------------------------
# Full CROPS dictionary (22 crops)
# ------------------------
CROPS = {
    "apple": {"img":"apple.jpg","description":"Apple requires cool climates and well-drained soil.","soil":"Loamy, rich in organic matter","temperature":"5–20°C","rainfall":"60–80 cm","pH":"5.5–6.5","benefits":"Rich in vitamins and antioxidants.","notes":"Requires chilling hours in winter."},
    "banana": {"img":"banana.jpg","description":"Banana grows well in humid tropical climates.","soil":"Rich loamy soil","temperature":"20–35°C","rainfall":"100–250 cm","pH":"6.0–7.0","benefits":"High-energy fruit.","notes":"Requires regular irrigation."},
    "blackgram": {"img":"blackgram.jpg","description":"Blackgram is an important pulse crop.","soil":"Sandy loam to clay soil","temperature":"25–35°C","rainfall":"60–90 cm","pH":"6.0–7.5","benefits":"Rich in protein.","notes":"Grows well during Kharif season."},
    "chickpea": {"img":"chickpea.jpg","description":"Chickpea grows best in dry, cool climates.","soil":"Well-drained loam","temperature":"10–25°C","rainfall":"40–60 cm","pH":"6.0–7.0","benefits":"Rich in protein and fiber.","notes":"Does not tolerate waterlogging."},
    "coconut": {"img":"coconut.jpg","description":"Coconut thrives in coastal and humid climates.","soil":"Sandy loam","temperature":"20–30°C","rainfall":"150–250 cm","pH":"5.0–8.0","benefits":"Used for oil, water, food.","notes":"Requires abundant sunlight."},
    "coffee": {"img":"coffee.jpg","description":"Coffee requires cool, shaded, high-altitude conditions.","soil":"Well-drained loamy soil","temperature":"15–24°C","rainfall":"150–250 cm","pH":"6.0–6.5","benefits":"Popular beverage crop.","notes":"Grows best under shade trees."},
    "cotton": {"img":"cotton.jpg","description":"Cotton grows in dry, warm climates with long frost-free periods.","soil":"Black cotton soil","temperature":"20–35°C","rainfall":"50–100 cm","pH":"5.5–7.0","benefits":"Used in textile industry.","notes":"Sensitive to excessive rainfall."},
    "grapes": {"img":"grapes.jpg","description":"Grapes need dry, warm climates with low humidity.","soil":"Sandy loam","temperature":"15–40°C","rainfall":"75–85 cm","pH":"6.5–7.5","benefits":"Used for juice, wine, raisins.","notes":"Requires pruning for good yield."},
    "jute": {"img":"jute.jpg","description":"Jute grows well in humid, warm climates.","soil":"Clayey loam","temperature":"25–35°C","rainfall":"150–200 cm","pH":"6.0–7.0","benefits":"Used to make strong fibers.","notes":"Requires standing water early on."},
    "kidneybeans": {"img":"kidneybeans.jpg","description":"Kidney beans grow in moderate climates.","soil":"Sandy loam","temperature":"15–25°C","rainfall":"60–100 cm","pH":"6.0–7.5","benefits":"High protein crop.","notes":"Requires well-drained soil."},
    "lentil": {"img":"lentil.jpg","description":"Lentil is a cool-season pulse crop.","soil":"Clay loam","temperature":"10–25°C","rainfall":"30–45 cm","pH":"6.0–8.0","benefits":"Rich in protein.","notes":"Drought tolerant."},
    "maize": {"img":"maize.jpg","description":"Maize thrives in warm climates and full sunlight.","soil":"Alluvial or black soil","temperature":"18–27°C","rainfall":"50–100 cm","pH":"5.5–7.0","benefits":"Used for food, fodder, biofuel.","notes":"Requires irrigation at tasseling stage."},
    "mango": {"img":"mango.jpg","description":"Mango grows in warm tropical areas.","soil":"Well-drained loam","temperature":"24–30°C","rainfall":"75–250 cm","pH":"5.5–7.0","benefits":"Highly nutritious fruit.","notes":"Prefers dry weather during flowering."},
    "mothbeans": {"img":"mothbeans.jpg","description":"Moth beans grow in arid regions.","soil":"Sandy soil","temperature":"25–35°C","rainfall":"20–30 cm","pH":"7.0–8.0","benefits":"Rich in protein.","notes":"Extremely drought resistant."},
    "mungbean": {"img":"mungbean.jpg","description":"Mungbean is a popular pulse crop.","soil":"Sandy loam","temperature":"25–30°C","rainfall":"50–75 cm","pH":"6.0–7.5","benefits":"Used for dal and sprouts.","notes":"Short-duration crop."},
    "muskmelon": {"img":"muskmelon.jpg","description":"Muskmelon grows in hot climates.","soil":"Sandy loam","temperature":"25–35°C","rainfall":"40–60 cm","pH":"6.0–7.5","benefits":"Sweet, hydrating fruit.","notes":"Requires good drainage."},
    "orange": {"img":"orange.jpg","description":"Orange requires subtropical climate.","soil":"Loamy soil","temperature":"15–30°C","rainfall":"75–100 cm","pH":"5.5–6.5","benefits":"Rich in vitamin C.","notes":"Requires good irrigation."},
    "papaya": {"img":"papaya.jpg","description":"Papaya grows well in tropical and subtropical climates.","soil":"Well-drained loam","temperature":"25–35°C","rainfall":"100–150 cm","pH":"6.5–7.0","benefits":"Rich in enzymes.","notes":"Fast-growing fruit crop."},
    "pigeonpeas": {"img":"pigeonpeas.jpg","description":"Pigeonpeas grow in semi-arid climates.","soil":"Loamy soil","temperature":"20–35°C","rainfall":"60–100 cm","pH":"6.0–7.0","benefits":"Important pulse crop.","notes":"Deep-rooted and drought-resistant."},
    "pomegranate": {"img":"pomegranate.jpg","description":"Pomegranate thrives in dry, hot climates.","soil":"Loamy or sandy loam","temperature":"25–35°C","rainfall":"50–100 cm","pH":"5.5–7.5","benefits":"Highly nutritious fruit.","notes":"Requires dry climate for fruiting."},
    "rice": {"img":"rice.jpg","description":"Rice is a staple food crop grown mainly in warm, humid climates with abundant water.","soil":"Clayey loam or silt loam","temperature":"20–35°C","rainfall":"100–200 cm","pH":"5.0–6.5","benefits":"High-calorie grain used worldwide.","notes":"Requires standing water during growth."},
    "watermelon": {"img":"watermelon.jpg","description":"Watermelon grows best in hot, dry climates.","soil":"Sandy loam","temperature":"25–35°C","rainfall":"40–60 cm","pH":"6.0–7.5","benefits":"Hydrating summer fruit.","notes":"Requires well-drained soil."}
}

# ------------------------
# Routes
# ------------------------
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/register", methods=["GET","POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username","").strip()
        password = request.form.get("password","")
        if not username or not password:
            flash("Provide username and password", "warning")
            return redirect(url_for("register"))
        if User.query.filter_by(username=username).first():
            flash("User exists, login instead.", "warning")
            return redirect(url_for("login"))
        hashed = generate_password_hash(password)
        u = User(username=username, password=hashed)
        db.session.add(u)
        db.session.commit()
        flash("Registered. Please login.", "success")
        return redirect(url_for("login"))
    return render_template("register.html")

@app.route("/login", methods=["GET","POST"])
def login():
    if request.method == "POST":
        print("Login POST data:", dict(request.form))
        username = request.form.get("username","").strip()
        password = request.form.get("password","")
        next_from_form = request.form.get("next")
        next_from_query = request.args.get("next")
        if not username or not password:
            flash("Please enter username and password", "warning")
            return redirect(url_for("login"))
        user = User.query.filter_by(username=username).first()
        if not user or not check_password_hash(user.password, password):
            flash("Invalid username or password", "danger")
            return redirect(url_for("login"))
        login_user(user)
        flash("Logged in successfully.", "success")
        next_page = next_from_form or next_from_query
        if next_page and is_safe_url(next_page):
            return redirect(next_page)
        return redirect(url_for("crops"))
    return render_template("login.html")

@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash("Logged out.", "info")
    return redirect(url_for("home"))

@app.route("/crops")
@login_required
def crops():
    # add fallback images where missing
    crops_with_images = {}
    for k, v in CROPS.items():
        vcopy = v.copy()
        img = vcopy.get("img") or ""
        if not image_exists(img):
            vcopy["img"] = "crop.jpg"
        crops_with_images[k] = vcopy
    return render_template("crops.html", crops=crops_with_images)

@app.route("/predict", methods=["GET","POST"])
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
            flash("Please enter valid numeric values.", "danger")
            return redirect(url_for("predict"))
        if model is None:
            flash("Model not loaded.", "danger")
            return render_template("predict.html", prediction=None)
        features = np.array([[N,P,K,temperature,humidity,ph,rainfall]], dtype=float)
        raw_pred = model.predict(features)
        label = safe_inverse_transform(raw_pred)
        return render_template("predict.html", prediction=label)
    return render_template("predict.html", prediction=None)

# ------------------------
# Run
# ------------------------
if __name__ == "__main__":
    app.run(debug=True)

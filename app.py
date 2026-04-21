from flask import Flask, request, render_template, jsonify, session, redirect, url_for, flash
import pandas as pd
import random
from flask_sqlalchemy import SQLAlchemy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import os, json

app = Flask(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
app.secret_key = "shoprec_secret_2024_xkzp"
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get(
    "DATABASE_URL",
    "sqlite:///" + os.path.join(BASE_DIR, "ecom.db")
)
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)

# ── Database Models ────────────────────────────────────────────────────────────
class User(db.Model):
    id         = db.Column(db.Integer, primary_key=True)
    username   = db.Column(db.String(100), unique=True, nullable=False)
    email      = db.Column(db.String(100), unique=True, nullable=False)
    password   = db.Column(db.String(100), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    cart_items = db.relationship("CartItem",     backref="user", lazy=True, cascade="all,delete")
    wishlist   = db.relationship("WishlistItem", backref="user", lazy=True, cascade="all,delete")

class CartItem(db.Model):
    id         = db.Column(db.Integer, primary_key=True)
    user_id    = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    name       = db.Column(db.String(300), nullable=False)
    brand      = db.Column(db.String(100))
    price      = db.Column(db.Float)
    image_url  = db.Column(db.String(500))
    rating     = db.Column(db.Float)
    quantity   = db.Column(db.Integer, default=1)
    added_at   = db.Column(db.DateTime, default=datetime.utcnow)

class WishlistItem(db.Model):
    id         = db.Column(db.Integer, primary_key=True)
    user_id    = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    name       = db.Column(db.String(300), nullable=False)
    brand      = db.Column(db.String(100))
    price      = db.Column(db.Float)
    image_url  = db.Column(db.String(500))
    rating     = db.Column(db.Float)
    added_at   = db.Column(db.DateTime, default=datetime.utcnow)

class Order(db.Model):
    id             = db.Column(db.Integer, primary_key=True)
    user_id        = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    full_name      = db.Column(db.String(100), nullable=False)
    mobile         = db.Column(db.String(15),  nullable=False)
    email          = db.Column(db.String(100), nullable=False)
    address1       = db.Column(db.String(200), nullable=False)
    address2       = db.Column(db.String(200))
    city           = db.Column(db.String(100), nullable=False)
    state          = db.Column(db.String(100), nullable=False)
    pincode        = db.Column(db.String(10),  nullable=False)
    landmark       = db.Column(db.String(150))
    payment_method = db.Column(db.String(30),  default='cod')
    total_amount   = db.Column(db.Float,       default=0)
    status         = db.Column(db.String(50),  default='Confirmed')
    placed_at      = db.Column(db.DateTime,    default=datetime.utcnow)
    cancelled_at   = db.Column(db.DateTime)
    cancel_reason  = db.Column(db.String(200))
    items_snapshot = db.Column(db.Text)

with app.app_context():
    db.create_all()

# ── Load & prepare data ────────────────────────────────────────────────────────
trending_products = pd.read_csv(os.path.join(BASE_DIR, "models", "trending_products.csv"))
train_data        = pd.read_csv(os.path.join(BASE_DIR, "models", "clean_data.csv"))

def first_url(v):
    if pd.isna(v): return ""
    return str(v).split("|")[0].strip()

trending_products["ImageURL"] = trending_products["ImageURL"].apply(first_url)
train_data["ImageURL"]        = train_data["ImageURL"].apply(first_url)
train_data["Tags"]            = train_data["Tags"].fillna("")
train_data["Brand"]           = train_data["Brand"].fillna("Unknown")
train_data["Rating"]          = pd.to_numeric(train_data["Rating"],       errors="coerce").fillna(0)
train_data["ReviewCount"]     = pd.to_numeric(train_data["ReviewCount"],  errors="coerce").fillna(0)

print("Building TF-IDF matrix …")
_tfidf = TfidfVectorizer(stop_words="english")
_mat   = _tfidf.fit_transform(train_data["Tags"])
print(f"Done — {len(train_data)} products indexed.")

PRICES = [999, 1299, 1499, 1799, 2499, 3299, 4999, 599, 799, 2199]

# ── Helpers ────────────────────────────────────────────────────────────────────
def truncate(text, n):
    t = str(text)
    return t[:n] + "…" if len(t) > n else t

def current_user():
    uid = session.get("user_id")
    return User.query.get(uid) if uid else None

def cart_count():
    u = current_user()
    if not u: return 0
    return sum(i.quantity for i in u.cart_items)

def recommend(item_name, top_n=8):
    mask = train_data["Name"].str.lower() == item_name.lower()
    if not mask.any():
        mask = train_data["Name"].str.lower().str.contains(item_name.lower(), na=False)
    if not mask.any():
        return []
    idx     = train_data[mask].index[0]
    sims    = cosine_similarity(_mat[idx], _mat).flatten()
    ranking = sorted(enumerate(sims), key=lambda x: x[1], reverse=True)
    top     = [x for x in ranking if x[0] != idx][:top_n]
    out = []
    for i, score in top:
        r = train_data.iloc[i]
        out.append({
            "name":         str(r["Name"]),
            "brand":        str(r["Brand"]),
            "rating":       float(r["Rating"]),
            "review_count": int(r["ReviewCount"]),
            "image_url":    str(r["ImageURL"]),
            "score":        round(float(score), 4),
            "price":        random.choice(PRICES),
        })
    return out

def base_ctx():
    u = current_user()
    return {"current_user": u, "cart_count": cart_count()}

def trending_ctx():
    ctx = base_ctx()
    ctx["trending_products"] = trending_products.head(8).reset_index(drop=True)
    ctx["truncate"]          = truncate
    ctx["random_price"]      = random.choice(PRICES)
    return ctx

# ── Auth Routes ────────────────────────────────────────────────────────────────
@app.route("/signup", methods=["GET","POST"])
def signup():
    if request.method == "POST":
        username = request.form["username"].strip()
        email    = request.form["email"].strip()
        password = request.form["password"]
        if User.query.filter_by(username=username).first():
            flash("Username already taken.", "danger")
            return redirect(url_for("signup"))
        if User.query.filter_by(email=email).first():
            flash("Email already registered.", "danger")
            return redirect(url_for("signup"))
        user = User(username=username, email=email, password=password)
        db.session.add(user)
        db.session.commit()
        session["user_id"] = user.id
        flash(f"Welcome, {username}! Account created.", "success")
        return redirect(url_for("index"))
    return render_template("signup.html", **base_ctx())

@app.route("/signin", methods=["GET","POST"])
def signin():
    if request.method == "POST":
        username = request.form["username"].strip()
        password = request.form["password"]
        user = User.query.filter_by(username=username, password=password).first()
        if user:
            session["user_id"] = user.id
            flash(f"Welcome back, {user.username}!", "success")
            return redirect(url_for("index"))
        flash("Invalid username or password.", "danger")
        return redirect(url_for("signin"))
    return render_template("signin.html", **base_ctx())

@app.route("/signout")
def signout():
    session.clear()
    flash("You have been signed out.", "info")
    return redirect(url_for("index"))

# ── Cart Routes ────────────────────────────────────────────────────────────────
@app.route("/cart")
def cart():
    u = current_user()
    if not u:
        flash("Please sign in to view your cart.", "warning")
        return redirect(url_for("signin"))
    items = u.cart_items
    total = sum(i.price * i.quantity for i in items)
    return render_template("cart.html", items=items, total=total, **base_ctx())

@app.route("/api/cart/add", methods=["POST"])
def api_cart_add():
    u = current_user()
    if not u:
        return jsonify({"error": "Please sign in first.", "redirect": "/signin"}), 401
    d = request.get_json()
    existing = CartItem.query.filter_by(user_id=u.id, name=d["name"]).first()
    if existing:
        existing.quantity += 1
    else:
        db.session.add(CartItem(
            user_id=u.id, name=d["name"], brand=d.get("brand",""),
            price=float(d.get("price", 49)), image_url=d.get("image_url",""),
            rating=float(d.get("rating", 0))
        ))
    db.session.commit()
    return jsonify({"success": True, "cart_count": cart_count(), "message": f"'{d['name'][:40]}...' added to cart!"})

@app.route("/api/cart/remove", methods=["POST"])
def api_cart_remove():
    u = current_user()
    if not u:
        return jsonify({"error": "Not signed in"}), 401
    d    = request.get_json()
    item = CartItem.query.filter_by(id=d["id"], user_id=u.id).first()
    if item:
        db.session.delete(item)
        db.session.commit()
    total = sum(i.price * i.quantity for i in u.cart_items)
    return jsonify({"success": True, "cart_count": cart_count(), "total": round(total, 2)})

@app.route("/api/cart/update", methods=["POST"])
def api_cart_update():
    u = current_user()
    if not u:
        return jsonify({"error": "Not signed in"}), 401
    d    = request.get_json()
    item = CartItem.query.filter_by(id=d["id"], user_id=u.id).first()
    if item:
        item.quantity = max(1, int(d["quantity"]))
        db.session.commit()
    total = sum(i.price * i.quantity for i in u.cart_items)
    return jsonify({"success": True, "cart_count": cart_count(), "total": round(total, 2)})

@app.route("/api/cart/clear", methods=["POST"])
def api_cart_clear():
    u = current_user()
    if not u:
        return jsonify({"error": "Not signed in"}), 401
    CartItem.query.filter_by(user_id=u.id).delete()
    db.session.commit()
    return jsonify({"success": True, "cart_count": 0})

# ── Wishlist Routes ────────────────────────────────────────────────────────────
@app.route("/wishlist")
def wishlist():
    u = current_user()
    if not u:
        flash("Please sign in to view your wishlist.", "warning")
        return redirect(url_for("signin"))
    return render_template("wishlist.html", items=u.wishlist, **base_ctx())

@app.route("/api/wishlist/add", methods=["POST"])
def api_wishlist_add():
    u = current_user()
    if not u:
        return jsonify({"error": "Please sign in first.", "redirect": "/signin"}), 401
    d = request.get_json()
    if WishlistItem.query.filter_by(user_id=u.id, name=d["name"]).first():
        return jsonify({"success": True, "message": "Already in your wishlist!", "already": True})
    db.session.add(WishlistItem(
        user_id=u.id, name=d["name"], brand=d.get("brand",""),
        price=float(d.get("price", 49)), image_url=d.get("image_url",""),
        rating=float(d.get("rating", 0))
    ))
    db.session.commit()
    return jsonify({"success": True, "message": "Added to wishlist!"})

@app.route("/api/wishlist/remove", methods=["POST"])
def api_wishlist_remove():
    u = current_user()
    if not u:
        return jsonify({"error": "Not signed in"}), 401
    d    = request.get_json()
    item = WishlistItem.query.filter_by(id=d["id"], user_id=u.id).first()
    if item:
        db.session.delete(item)
        db.session.commit()
    return jsonify({"success": True})

# ── Checkout Routes ────────────────────────────────────────────────────────────
@app.route("/checkout")
def checkout():
    u = current_user()
    if not u:
        flash("Please sign in to checkout.", "warning")
        return redirect(url_for("signin"))
    items = u.cart_items
    if not items:
        flash("Your cart is empty!", "warning")
        return redirect(url_for("cart"))
    total    = sum(i.price * i.quantity for i in items)
    delivery = 0 if total >= 999 else 49
    discount = round(total * 0.1)
    final    = round(total - discount + delivery)
    return render_template("checkout.html", items=items, total=total,
                           delivery=delivery, discount=discount, final=final, **base_ctx())

@app.route("/place_order", methods=["POST"])
def place_order():
    u = current_user()
    if not u:
        return redirect(url_for("signin"))
    items = u.cart_items
    if not items:
        flash("Your cart is empty!", "warning")
        return redirect(url_for("cart"))
    total    = sum(i.price * i.quantity for i in items)
    delivery = 0 if total >= 999 else 49
    discount = round(total * 0.1)
    final    = round(total - discount + delivery)
    snapshot = json.dumps([{"name": i.name, "brand": i.brand, "price": i.price,
                             "qty": i.quantity, "image_url": i.image_url} for i in items])
    order = Order(
        user_id        = u.id,
        full_name      = request.form["full_name"].strip(),
        mobile         = request.form["mobile"].strip(),
        email          = request.form["email"].strip(),
        address1       = request.form["address1"].strip(),
        address2       = request.form.get("address2", "").strip(),
        city           = request.form["city"].strip(),
        state          = request.form["state"].strip(),
        pincode        = request.form["pincode"].strip(),
        landmark       = request.form.get("landmark", "").strip(),
        payment_method = request.form.get("payment_method", "cod"),
        total_amount   = final,
        items_snapshot = snapshot,
    )
    db.session.add(order)
    CartItem.query.filter_by(user_id=u.id).delete()
    db.session.commit()
    return redirect(url_for("order_confirmed", order_id=order.id))

@app.route("/order_confirmed/<int:order_id>")
def order_confirmed(order_id):
    u = current_user()
    if not u:
        return redirect(url_for("signin"))
    order = Order.query.filter_by(id=order_id, user_id=u.id).first()
    if not order:
        flash("Order not found.", "danger")
        return redirect(url_for("index"))
    items = json.loads(order.items_snapshot or "[]")
    return render_template("order_confirmed.html", order=order, items=items, **base_ctx())

@app.route("/my_orders")
def my_orders():
    u = current_user()
    if not u:
        flash("Please sign in to view your orders.", "warning")
        return redirect(url_for("signin"))
    orders = Order.query.filter_by(user_id=u.id).order_by(Order.placed_at.desc()).all()
    return render_template("my_orders.html", orders=orders, **base_ctx())

@app.route("/cancel_order/<int:order_id>", methods=["POST"])
def cancel_order(order_id):
    u = current_user()
    if not u:
        return redirect(url_for("signin"))
    order = Order.query.filter_by(id=order_id, user_id=u.id).first()
    if not order:
        flash("Order not found.", "danger")
        return redirect(url_for("my_orders"))
    if order.status in ("Cancelled", "Delivered"):
        flash(f"Order cannot be cancelled.", "warning")
        return redirect(url_for("my_orders"))
    order.status       = "Cancelled"
    order.cancelled_at = datetime.utcnow()
    order.cancel_reason= request.form.get("reason", "Cancelled by customer")
    db.session.commit()
    flash(f"Order #{ order.id:04d} has been cancelled.", "info")
    return redirect(url_for("my_orders"))

@app.template_filter("from_json")
def from_json_filter(value):
    try: return json.loads(value or "[]")
    except: return []

# ── Page Routes ────────────────────────────────────────────────────────────────
@app.route("/")
@app.route("/index")
def index():
    return render_template("index.html", **trending_ctx())

@app.route("/main")
def main():
    return render_template("main.html", **base_ctx())

# ── API: Autocomplete ──────────────────────────────────────────────────────────
@app.route("/api/products")
def api_products():
    q     = request.args.get("q","").lower()
    names = train_data["Name"].dropna().unique().tolist()
    if q:
        names = [n for n in names if q in n.lower()][:40]
    else:
        names = names[:40]
    return jsonify(names)

# ── API: Recommend ─────────────────────────────────────────────────────────────
@app.route("/api/recommend", methods=["POST"])
def api_recommend():
    d     = request.get_json()
    prod  = (d.get("prod") or "").strip()
    top_n = min(int(d.get("nbr", 8)), 20)
    if not prod:
        return jsonify({"error": "Please enter a product name."}), 400
    recs = recommend(prod, top_n)
    if not recs:
        return jsonify({"error": f"No recommendations found for '{prod}'. Try a shorter keyword."}), 404
    return jsonify({"query": prod, "count": len(recs), "results": recs})

if __name__ == "__main__":
    app.run(debug=False)

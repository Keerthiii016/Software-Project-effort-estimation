

from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
import joblib
import numpy as np
import pandas as pd
import os
from flask_sqlalchemy import SQLAlchemy
import smtplib

import random
import string
from dotenv import load_dotenv
from email.message import EmailMessage
from werkzeug.security import generate_password_hash, check_password_hash
import re


app = Flask(__name__)
app.secret_key = "your_secret_key"

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html")

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    otp = db.Column(db.String(10))# Added email column for OTP verification



load_dotenv()  # Load from .env

sender_email = os.getenv('EMAIL_USER')
sender_password = os.getenv('EMAIL_PASS')


# Generate OTP
def generate_otp():
    otp = ''.join(random.choices(string.digits, k=6))  # Generate a 6-digit OTP
    return otp


# Send OTP email function



def send_otp_email(to_email, otp):
    sender_email = os.getenv('EMAIL_USER')
    sender_password = os.getenv('EMAIL_PASS')

    msg = EmailMessage()
    msg.set_content(f'Your OTP is: {otp}')
    msg['Subject'] = 'OTP Verification - Effort Estimation'
    msg['From'] = sender_email
    msg['To'] = to_email

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(sender_email, sender_password)
            smtp.send_message(msg)
            print("OTP sent successfully.")
    except Exception as e:
        print("Error sending OTP:", e)

# Load the trained model and preprocessing objects
model = joblib.load("random_forest_model.pkl")
scaler = joblib.load("scaler.pkl")
selected_features = joblib.load("selected_features.pkl")

# User management
users = {"admin": "Admin@123"}  # Updated admin password

otp_storage = {}  # Store OTPs temporarily for verification


def preprocess_input(user_input):
    try:
        input_df = pd.DataFrame([user_input])
        user_scaled = scaler.transform(input_df[selected_features])
        return user_scaled
    except Exception as e:
        return f"Preprocessing Error: {str(e)}"


def classify_level(value):
    levels = {
        1: "More than 4 years",
        2: "2 to 3 years",
        3: "1 year to 2 years",
        4: "6 months - 1 year",
        5: "First time in this project"
    }
    return levels.get(value, "Unknown")


prod_table = {
    "More than 4 years": 50,
    "2 to 3 years": 25,
    "1 year to 2 years": 13,
    "6 months - 1 year": 7,
    "First time in this project": 4,
}




@app.route("/login", methods=["POST"])
def login():
    username = request.form["username"]
    password = request.form["password"]

    user = User.query.filter_by(username=username).first()

    if user and check_password_hash(user.password, password):
        session["user"] = user.username
        return redirect(url_for("home"))
    else:
        flash("Invalid username or password. Try again.", "danger")
        return redirect(url_for("home"))


def is_valid_email(email):
    pattern = r"^[\w\.-]+@[\w\.-]+\.\w+$"
    return re.match(pattern, email)



@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        new_username = request.form["new_username"]
        new_password = request.form["new_password"]
        email = request.form["email"]

        # Validate email format

        if not is_valid_email(email):
            flash("Invalid email format. Please enter a valid email.", "signup_error")
            return redirect(url_for("index"))  # Redirect back to index.html

            # Check if username or email already exists
        if User.query.filter_by(username=new_username).first():
            flash("Username already exists. Try a different one.", "signup_error")
            return redirect(url_for("index"))

        if User.query.filter_by(email=email).first():
            flash("Email already exists. Try a different one.", "signup_error")
            return redirect(url_for("index"))

        # Generate OTP & store signup details
        otp = generate_otp()
        otp_storage[email] = otp
        send_otp_email(email, otp)

        session["pending_user"] = {
            "username": new_username,
            "password": generate_password_hash(new_password),
            "email": email,
            "otp": otp
        }

        flash("OTP sent to your email. Please check your inbox.", "info")
        return redirect(url_for("verify_otp"))

    return redirect(url_for("index"))  # Ensure signup redirects back to index




def is_valid_email(email):
    """Simple email validation using regex"""
    import re
    pattern = r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'
    return bool(re.match(pattern, email))




@app.route("/verify_otp", methods=["GET", "POST"])
def verify_otp():
    if request.method == "POST":
        entered_otp = request.form["otp"]
        pending_user = session.get("pending_user")  # fetch the stored data

        if pending_user and entered_otp == otp_storage.get(pending_user["email"]):
            # Create and store the new user
            new_user = User(
                username=pending_user["username"],
                password=pending_user["password"],
                email=pending_user["email"],
                otp=entered_otp
            )
            db.session.add(new_user)
            db.session.commit()

            # Clean up session and OTP storage
            session.pop("pending_user", None)
            otp_storage.pop(pending_user["email"], None)

            flash("Signup successful. Please log in.", "success")
            return redirect(url_for("login"))
        else:
            flash("Invalid OTP or expired OTP. Please try again.", "danger")

    return render_template("verify_otp.html")



@app.route("/home")
def home():

    if "user" not in session:
        flash("Please log in to continue.", "warning")
        return redirect(url_for("login"))
    return render_template("home.html")



@app.route("/predict", methods=["POST"])
def predict():
    if "user" not in session:
        return jsonify({"error": "Unauthorized"}), 401
    try:
        data = request.get_json() if request.is_json else request.form.to_dict()
        print("Received Data:", data)

        if not data:
            return jsonify({"error": "No data received"}), 400

        if "formula_submit" in data:
            try:
                object_points = float(data.get("Object_points", 0))
                software_reuse = float(data.get("Degree_of_software_reuse_", 0))
                exp_level = classify_level(int(data.get("Programmers_experience", 0)))
                cap_level = classify_level(int(data.get("Programmers_capability", 0)))
                avg_salary = float(data.get("Avg_salary", 0))
                team_size = int(data.get("Team_size", 1))
                dedicated_team_members = int(data.get("Dedicated_team_members", 1))
                working_hours = float(data.get("Working_hours", 8))
                NOP = (object_points * (100 - software_reuse)) / 100
                hr = int(20 * working_hours)

                if exp_level == cap_level:
                    PROD = prod_table[exp_level]  # Directly use the value from the table
                else:
                    PROD = (prod_table[exp_level] + prod_table[cap_level]) / 2  # Take the average

                formula_effort = max(1, int(NOP / PROD) * hr)


                total_working_hours_per_year = working_hours * 5 * 52

                # Calculate cost using the dynamically computed hourly rate
                formula_cost = round((avg_salary / total_working_hours_per_year) * formula_effort, 2)

                denominator = (dedicated_team_members + ((team_size - dedicated_team_members) * 0.5)) * (
                        working_hours * 22)

                if denominator == 0:
                    return jsonify({"error": "Invalid team structure leading to zero division"}), 400
                formula_duration = round(formula_effort / denominator, 2)  # Instead of int()

                return jsonify({
                    "formula_effort": round(formula_effort, 2),
                    "formula_duration": round(formula_duration, 2),
                    "formula_cost": round(formula_cost, 2)
                })
            except Exception as e:
                return jsonify({"error": f"Formula calculation error: {str(e)}"}), 400

        elif "model_submit" in data:
            try:
                user_input = {key: float(data.get(key, 0)) for key in selected_features}
                avg_salary = float(data.get("Avg_salary", 0))  # Ensure avg_salary is retrieved

                working_hours = float(data.get("Working_hours", 8))  # Default: 8
                team_size = int(data.get("Team_size", 1))
                dedicated_team_members = int(data.get("Dedicated_team_members", 1))  #

                user_scaled = preprocess_input(user_input)
                if isinstance(user_scaled, str):
                    return jsonify({"error": user_scaled}), 400
                model_effort = round(float(model.predict(user_scaled)[0]), 2)

                # Convert to integer, divide by 10, and remove last digit
                adjusted_effort = int(model_effort // 10)  # Floor division to remove last digit

                # Compute total working hours per year based on user input
                total_working_hours_per_year = working_hours * 5 * 52

                # Calculate cost using the dynamically computed hourly rate
                model_cost = round((avg_salary / total_working_hours_per_year) * model_effort, 2)

                denominator = (dedicated_team_members + ((team_size - dedicated_team_members) * 0.5)) * (
                        working_hours * 22)

                if denominator == 0:
                    return jsonify({"error": "Invalid team structure leading to zero division"}), 400

                model_duration = int(adjusted_effort / denominator)  # Corrected duration calculation

                return jsonify({
                    "model_effort": adjusted_effort,
                    "model_duration": round(model_duration, 2),  # Updated effort calculation
                    "model_cost": model_cost
                })


            except Exception as e:
                return jsonify({"error": f"Model prediction error: {str(e)}"}), 500

        return jsonify({"error": "Invalid request, no recognized submit button"}), 400
    except Exception as e:
        print("Unexpected Error:", str(e))
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500


@app.route("/logout",methods=["GET","POST"])
def logout():
    session.clear()
    #session.pop("user", None)
    return redirect(url_for("index"))




if __name__ == "__main__":

    with app.app_context():
        db.create_all()
    app.run(debug=True)














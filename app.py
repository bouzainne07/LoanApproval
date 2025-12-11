from flask import Flask, render_template, request
import pandas as pd
import pickle

# Load trained model
with open("loan_model.pkl", "rb") as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    data = {
        "person_age": float(request.form["person_age"]),
        "person_gender": request.form["person_gender"],
        "person_education": request.form["person_education"],
        "person_income": float(request.form["person_income"]),
        "person_emp_exp": float(request.form["person_emp_exp"]),
        "person_home_ownership": request.form["person_home_ownership"],
        "loan_amnt": float(request.form["loan_amnt"]),
        "loan_intent": request.form["loan_intent"],
        "loan_int_rate": float(request.form["loan_int_rate"]),
        "loan_percent_income": float(request.form["loan_percent_income"]),
        "cb_person_cred_hist_length": float(request.form["cb_person_cred_hist_length"]),
        "credit_score": float(request.form["credit_score"]),
        "previous_loan_defaults_on_file": request.form["previous_loan_defaults_on_file"]
    }

    # Convert to DataFrame
    df = pd.DataFrame([data])

    # Predict
    proba = model.predict_proba(df)[0][1]   # probabilité d'être Approved
    prediction = "Approved" if proba >= 0.5 else "Rejected"


    return render_template("index.html", result=prediction, score=round(proba, 4))


if __name__ == "__main__":
    app.run(debug=True)

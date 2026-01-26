# Gender-> 1 Female 0 Male
#Churn -> 1 Yes 0 No
# scaler is exported as scaler.pkl
# model is exported as model.pkl
# order of the x -> 'age', 'gender', 'tenure', 'monthlycharges'


from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

scaler = joblib.load("scaler.pkl")
model = joblib.load("model.pkl")

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    show_balloons = False

    if request.method == "POST":
        age = float(request.form["age"])
        tenure = float(request.form["tenure"])
        monthlycharge = float(request.form["monthlycharge"])
        gender = int(request.form["gender"])

        # SAME ORDER AS STREAMLIT
        x = np.array([[age, gender, tenure, monthlycharge]])
        x_scaled = scaler.transform(x)

        result = model.predict(x_scaled)[0]
        prediction = "Yes" if result == 1 else "No"
        show_balloons = True

    return render_template(
        "index.html",
        prediction=prediction,
        show_balloons=show_balloons
    )

if __name__ == "__main__":
    app.run(debug=True)


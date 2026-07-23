from flask import Flask, render_template, request
from xgboost import XGBClassifier
import numpy as np

app = Flask(__name__)

model = XGBClassifier()
model.load_model("car_safety_model.json")


# Same encoding used while training the model
buying_map = {
    "low": 0,
    "med": 1,
    "high": 2,
    "vhigh": 3
}

maintenance_map = {
    "low": 0,
    "med": 1,
    "high": 2,
    "vhigh": 3
}

doors_map = {
    "2": 0,
    "3": 1,
    "4": 2,
    "5more": 3
}

persons_map = {
    "2": 0,
    "4": 1,
    "more": 2
}

luggage_map = {
    "small": 0,
    "med": 1,
    "big": 2
}

safety_map = {
    "low": 0,
    "med": 1,
    "high": 2
}


# Model output classes
result_map = {
    0: "Unacceptable",
    1: "Acceptable",
    2: "Good",
    3: "Very Good"
}


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    # Get values selected by user
    buying = request.form["buying"]
    maintenance = request.form["maintenance"]
    doors = request.form["doors"]
    persons = request.form["persons"]
    luggage = request.form["luggage"]
    safety = request.form["safety"]

    # Convert categorical values to numbers
    features = np.array([[
        buying_map[buying],
        maintenance_map[maintenance],
        doors_map[doors],
        persons_map[persons],
        luggage_map[luggage],
        safety_map[safety]
    ]])

    # Make prediction
    prediction = model.predict(features)[0]

    result = result_map[int(prediction)]

    # Different styling/message depending on result
    result_classes = {
        "Unacceptable": "unacceptable",
        "Acceptable": "acceptable",
        "Good": "good",
        "Very Good": "very-good"
    }

    messages = {
        "Unacceptable":
            "This vehicle does not meet the required evaluation standards.",

        "Acceptable":
            "This vehicle meets the basic evaluation requirements.",

        "Good":
            "This vehicle offers a good overall balance across the evaluated factors.",

        "Very Good":
            "This vehicle performs very well across the evaluated factors."
    }

    return render_template(
        "index.html",
        prediction=result,
        result_class=result_classes[result],
        result_message=messages[result],
        selected=request.form
    )


if __name__ == "__main__":
    app.run(debug=True)
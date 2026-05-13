
import pickle
import numpy as np

from flask import Flask, request, render_template

app = Flask(__name__)

# Load saved files
random_forest = pickle.load(
    open('random_forest.pkl', 'rb')
)

scaler = pickle.load(
    open('scaler.pkl', 'rb')
)

encoder = pickle.load(
    open('encoder.pkl', 'rb')
)

# Home page
@app.route('/')
def home():

    return render_template('interface.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():

    try:

        # Get input values
        data = [float(x) for x in request.form.values()]

        # Scale input
        final_input = scaler.transform(
            np.array(data).reshape(1, -1)
        )

        # Predict
        prediction = random_forest.predict(final_input)[0]

        # Decode label
        output = encoder.inverse_transform(
            [prediction]
        )[0]

        return render_template(
            'interface.html',
            prediction_text=f"The best suitable crop is: {output}"
        )

    except Exception as e:

        return render_template(
            'interface.html',
            prediction_text=f"Error: {str(e)}"
        )

if __name__ == "__main__":

    app.run(debug=True)
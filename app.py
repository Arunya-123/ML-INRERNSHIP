
import pickle
import numpy as np
from flask import Flask, request, render_template
app = Flask(__name__)

random_forest = pickle.load(open('random_forest.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
encoder = pickle.load(open('encoder.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('interface.html')
    
@app.route('/predict', methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    final_input = scaler.transform(np.array(data).reshape(1, -1))
    prediction = random_forest.predict(final_input)[0]
    output = encoder.inverse_transform([prediction])[0]
    return render_template('interface.html',prediction_text=f"The best suitable crop is: {output}")

if __name__ == "__main__":
    app.run(debug=True)

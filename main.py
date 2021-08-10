import numpy as np
import pickle
from flask import Flask, render_template


app = Flask(__name__)
model = pickle.load(open('Regression_Model.pkl', 'rb'))


@app.route("/")
def salary():
    return render_template("index.html")


@app.route("/predict", methods=['POST'])
def predict():
    float_feature = [float(x) for x in request.form.values()]
    final_feature = np.array(float_feature).reshape(-1, 1)
    prediction = model.predict(final_feature)

    return render_template("index.html", prediction_text="Your salary should be approximetely {}".format(prediction))


if __name__ == '__main__':
    app.run(debug=True)

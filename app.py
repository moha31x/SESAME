import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template
from sesame.config import Config

app = Flask(__name__)
model = pickle.load(open('model.pickle', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    ''' For rendering results '''
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template(
        'index.html', prediction_text='Production is estimated to be {} Metric Tonnes'.format(output)
    )


if __name__ == '__main__':
    app.run(debug=True)
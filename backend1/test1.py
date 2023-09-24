#USING ARIMA MODEL
from flask import Flask, request, render_template, jsonify, send_file
import io
from PIL import Image
from flask_cors import CORS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import os

app = Flask(__name__)
CORS(app)

accuracy = 0.0

@app.route('/', methods=['GET', 'POST'])
def hello_world():
    global accuracy
    if request.method == 'POST':
        def func(p, f, file_name):
            global accuracy
            data = pd.read_csv(file_name)
            
            # Data cleaning
            data = data.dropna()
            data['sales'] = data['sales'].astype(int)
            data['date'] = pd.to_datetime(data['date'])


            # Train the ARIMA model only once and store it in a variable
            model = sm.tsa.ARIMA(data['sales'], order=(1, 0, 1))
            model_fit = model.fit()

            train_data = data[:-2]
            test_data = data[-2:]
            print("Training")

            forecast = model_fit.forecast(steps=p)

            # Plotting
            print("Plotting")
            # Create a new datetime index for the forecasted period
            forecast_index = pd.date_range(start=data['date'].iloc[-1], periods=p+1, freq='MS')[1:]
            plt.figure(figsize=(10, 6))
            plt.plot(data['date'], data['sales'], label='Original Data')
            plt.plot(forecast_index, forecast, label='Forecast')
            plt.legend()
            plt.savefig('plottest.png')

            # Calculate accuracy
            y_true = test_data['sales'].values
            y_pred = forecast[:len(y_true)]
            accuracy = 100 - np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            accuracy = round(accuracy , 2)
            print(accuracy)
            print("Sending")
            # Send plot as response
            img = Image.open('plottest.png')
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='png')
            img_bytes.seek(0)
            return send_file(img_bytes, mimetype='image/png')

        val1 = request.form['period']
        val2 = request.form['range']
        print(val1)
        print(val2)

        if val1=="Yearly":
            f = 1
            p = int(val2) * 12
        elif val1=="Monthly":
            f = 1
            p = int(val2)
        elif val1=="Weekly":
            f = 7
            p = int(val2) * f
        elif val1=="Daily":
            f = 1
            p = int(val2)

        file = request.files['file']
        file_name = file.filename
        file.save(file_name)
        file_size = len(file.read())
        file_stats = os.stat(file_name)

        func(p, f, file_name)
        print("Sent")
        response_headers = {'Access-Control-Allow-Origin': '*'}
        response={'message': 'Success','accuracy':accuracy*100}
        return jsonify(response), 200, response_headers

    if request.method == 'GET':
        response = {'accuracy': accuracy * 100}
        return send_file('plottest.png', mimetype='image/png'), response
@app.route('/accuracy', methods=['GET'])
def get_accuracy():
    global accuracy
    response = {'accuracy': accuracy}
    return jsonify(response)  


if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask, request, render_template, jsonify, send_file,send_from_directory
import io
from PIL import Image
from flask_cors import CORS
import numpy as np
import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
import os

app = Flask(__name__)
CORS(app)

accuracy = 0.0
rmse = 0
mape = 0

@app.route('/', methods=['GET', 'POST'])
def hello_world():
    global accuracy
    global rmse
    global mape
    if request.method == 'POST':
        def func(p,f,file_name):
            global accuracy
            global rmse
            global mape
            data = pd.read_csv(file_name)
            #info
            print("Info")
            data.info()
            #display describe()
            print("Describe")
            print(data.describe())
            #shape
            print("Shape")
            print(data.shape)
            #null #drop
            print("Is null")
            print(data.isna())
            #null sum
            print("Sum of null values")
            print(data.isnull().sum())
            #replacing mean values for missing data
            data.sales.fillna(data.sales.mean(),inplace=True)
            #change to int
            data.sales = data.sales.astype(int)
            print("After Cleaning and type convertion")
            data.info()

            # Convert the date column to a datetime format  
            data['date'] = pd.to_datetime(data['date'])
            print("After changing to date time format")
            data.info()


            data = data.rename(columns={'date': 'ds', 'sales': 'y'})


            train_data = data[:-2]
            test_data = data[-2:]


            model = Prophet()
            model.fit(train_data)


            future = model.make_future_dataframe(periods=p, freq=f)
            forecast = model.predict(future)


            fig = model.plot(forecast,figsize=(8,5))

            plt.xlabel('Date')
            plt.ylabel('Sales')
            plt.title(label='Sales Forecast for '+str(p)+' '+f, fontweight=10, pad='0.0')

            plt.savefig('plot.png')
            forecast.to_csv('forecast.csv', index=False)

            y_true = test_data['y'].values
            y_pred = forecast['yhat'][-2:].values
            accuracy = 100 - np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            accuracy = round(accuracy , 2)
            print(accuracy)
            rmse = np.sqrt(np.mean(np.square(y_true - y_pred)))
            rmse = round(rmse,2)
            print(rmse)
            mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            mape = round(mape,2)
            print(mape)


            img = Image.open('plot.png')
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='png')
            img_bytes.seek(0)
            return send_file(img_bytes, mimetype='image/png')
        val1 = request.form['period']
        val2 = request.form['range']
        print(val1)
        print(val2)
        if val1=="Yearly":
            fre = "Y"
        elif val1=="Monthly":
            fre = "M"
        elif val1=="Weekly":
            fre = "W"
        elif val1=="Daily":
            fre = "D"

        per = int(val2)

        file = request.files['file']
        file_name = file.filename
        file.save(file_name)
        file_size = len(file.read())
        file_stats = os.stat(file_name)

        func(per,fre,file_name)
        response_headers = {'Access-Control-Allow-Origin': '*'}
        response = {'message': 'Success', 'accuracy': accuracy*100, 'rmse': rmse, 'mape': mape}
        return jsonify(response), 200, response_headers
    if request.method == 'GET':
        response = {'accuracy': accuracy * 100, 'rmse': rmse, 'mape': mape}
        return send_file('plot.png', mimetype='image/png'),response
@app.route('/accuracy', methods=['GET'])
def get_accuracy():
    global accuracy
    global rmse
    global mape
    response = {'accuracy': accuracy, 'rmse': rmse, 'mape': mape}
    return jsonify(response)   



if __name__ == '__main__':
    app.run(debug=True)





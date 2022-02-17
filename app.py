import numpy as np
from flask import Flask, request, render_template
from joblib import load, dump
import pandas as pd
from feature_engine.datetime import DatetimeFeatures


app = Flask(__name__)

model = load("./models/model2_best.joblib")


@app.route("/test")
def test():
    return render_template("test.html")


@app.route('/')
def home():
    return render_template('index.html')


@app.route("/result")
def result():
    return render_template("result.html")


@app.route('/prediction')
def prediction():
    return render_template('prediction.html')


@app.route('/predict', methods=['POST', 'GET'])
def predict():

    data = request.form.to_dict()
    df = pd.DataFrame([data])
    df.to_csv("./features.csv", index=False)

    dtf = DatetimeFeatures(variables='saledate',
                           features_to_extract=['month', 'year', 'day_of_week', 'weekend', 'month_start',
                                                'month_end', 'quarter_start', 'quarter_end'],
                           drop_original=True)
    df = dtf.fit_transform(df)
    model = load("./models/model2_best.joblib")
    pred = model.predict(df)
    price = round(pred[0], 2)
   #  int_features = [x for x in request.form.values()]
   # preprocessed_features = [np.array(int_features)]
  #   features = pd.DataFrame(data)
  #  features.to_csv("./features.csv")
    # need to process input features in order to tranform into correct formats/encoding
    # convert sale date field into desired fields using date part parser
    # ensure all fields have correct type

    # final_features = None
    # prediction = model.predict(final_features)
    return render_template("result.html", prediction_text="The predicted price of the machinery  is: ${:,.2f}".format(price), submission=data)
#    return render_template('index.html',
 #                          prediction_text='The predicted price of the machinery is ${}'.format(
  #                             prediction),
   #                        )


if __name__ == "__main__":
    app.run(debug=False)

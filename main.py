import sys
import threading
import requests

from datetime import datetime, timedelta
from time import sleep

import numpy as np
import pyESN
from Historic_Crypto import HistoricalData


# date formatting in ISO 8601
def string_format(date):
    return f"{date.year}-{date.month}-{date.day}-{date.hour}-{date.minute}"


# countdown for waiting until start of minute
def _countdown_wait(time):
    while datetime.now().second != 0:
        print(f"Countdown: {60 - datetime.now().second}")
        sleep(time)


# network training
def train(network_obj, data_obj):
    prediction_training = network_obj.fit(np.ones(training_set_size), data_obj[0:training_set_size])
    return network_obj.predict(np.ones(1))


# data stream thread
def _data_thread(data_obj, network_obj, prediction):
    # infinite loop
    while True:
        # sleep so as not to initiate multiple json requests on minute
        sleep(2)

        # recursive loop
        while True:

            # logic to check for price
            if datetime.now().second == 0:

                # gather price
                _api_request = requests.get(request_url).json()

                price = float(_api_request.get('price'))
                data_obj[data_obj.size-1] = price

                # append data with current price, keeps same size
                data_obj = np.insert(data_obj[1:], data_obj.size - 1, price)

                print(f"current price: {price}\n")
                print("training network...")
                prediction = train(network_obj=network_obj, data_obj=data_obj)

                print(f"current prediction: {prediction}\n")

                break


# purchasing thread
def _purchasing_thread(data_obj, goal, prediction):
    global start
    while True:
        holding = False

        if prediction / data_obj[data_obj.size - 1] >= goal:
            holding = True

            # get price on buy
            start = data_obj[data_obj.size - 1]

        while holding:
            if prediction / data_obj[data_obj.size - 1] < goal or start > data_obj[data_obj.size - 1]:
                print(f"Sell at: {data_obj[data_obj.size - 1]}")
                break

            elif prediction / data_obj[data_obj.size - 1] > goal or start > data_obj[data_obj.size - 1]:
                print(f"Buy at: {data_obj[data_obj.size - 1]}")


# parameter initialization
training_set_size = 1500
currency = 'ADA'
granularity = 60
goal_increase = 1.02  # percentage increase

request_url = 'https://api.pro.coinbase.com/products/ADA-USD/ticker'

# countdown until even minute
_countdown_wait(1)


# time
end_date = datetime.now()
start_date = end_date - timedelta(days=training_set_size / ((60 / (granularity / 60)) * 24))

# historical data
raw_data = HistoricalData(ticker=f'{currency}-USD',
                          granularity=granularity,
                          start_date=string_format(start_date),
                          end_date=string_format(end_date)).retrieve_data()


_data_arr = np.zeros(shape=len(raw_data))

# initializing intermediate array with the closing prices of raw historical data
for i in np.arange(0, len(raw_data)):
    _data_arr[i] = raw_data['close'][i]


if _data_arr.size != training_set_size:
    if _data_arr.size < training_set_size:
        sys.exit("Error occurred in historical data input. Program terminating...")
    else:
        for i in np.arange(0, _data_arr.size - training_set_size):
            data = np.delete(_data_arr, np.arange(0, _data_arr.size - training_set_size), None)

del _data_arr


# parameter initialization for neural network
n_reservoir = 500
sparsity = 0.2
rand_seed = 34
spectral_radius = 0.95
noise = .0003

# network
esn = pyESN.ESN(n_inputs=1,
                n_outputs=1,
                n_reservoir=n_reservoir,
                sparsity=sparsity,
                random_state=rand_seed,
                spectral_radius=spectral_radius,
                noise=noise)

print(f"dataset size: {data.size}")
print(f"dataset has elements: {data}")


_prediction = 0.0

# data stream thread
data_thread = threading.Thread(target=_data_thread,
                               args=(data, esn, _prediction),
                               daemon=False)

data_thread.start()

# purchasing thread
purchasing_thread = threading.Thread(target=_purchasing_thread,
                                     args=(data, goal_increase, _prediction),
                                     daemon=False)

purchasing_thread.start()

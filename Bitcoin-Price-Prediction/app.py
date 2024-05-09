from flask import Flask, render_template, request
import pickle
from flask.templating import _default_template_ctx_processor
import numpy as np
import pandas as pd

filename = 'rf_random.pkl'
model = pickle.load(open(filename, 'rb'))

app =  Flask(__name__)

@app.route('/')
def home():
	return render_template('main.html')

@app.route('/predict', methods=['POST'])

def main():
    #temp_array =list()
    if request.method == 'POST':
        sentinusd90momUSD = float(request.form['sentinusd90momUSD'])
        hashrate90mom = float(request.form['hashrate90mom'])
        difficulty90mom = float(request.form['difficulty90mom'])
        activeaddresses7std = float(request.form['activeaddresses7std'])
        difficulty7std = float(request.form['difficulty7std'])
        price14momUSD= float(request.form['price14momUSD'])
        sentinusdUSD= float(request.form['sentinusdUSD'])
        transactionvalue3stdUSD= float(request.form['transactionvalue3stdUSD'])
        activeaddresses3std= float(request.form['activeaddresses3std'])
        transactions3std= float(request.form['transactions3std'])
        price30momUSD= float(request.form['price30momUSD'])
        fee_to_reward3stdUSD= float(request.form['fee_to_reward3stdUSD'])
        mining_profitability90trx= float(request.form['mining_profitability90trx'])
        sentinusd30momUSD= float(request.form['sentinusd30momUSD'])
        transactionvalue30momUSD= float(request.form['transactionvalue30momUSD'])
        transactions= float(request.form['transactions'])
        difficulty= float(request.form['difficulty'])
        difficulty14std= float(request.form['difficulty14std'])
        difficulty30mom= float(request.form['difficulty30mom'])
        mining_profitability30trx= float(request.form['mining_profitability30trx'])
        Change= float(request.form['Change'])
        expanding_mean= float(request.form['expanding_mean'])
        lag_1= float(request.form['lag_1'])
        lag_2= float(request.form['lag_2'])
        lag_3= float(request.form['lag_3'])
        lag_4= float(request.form['lag_4'])
        lag_5= float(request.form['lag_5'])
        lag_6= float(request.form['lag_6'])
        lag_7= float(request.form['lag_7'])
        Return= float(request.form['Return'])
        Mean= float(request.form['Mean'])
        difference= float(request.form['difference'])
        d30day_WMA= float(request.form['30day_WMA'])
        d30_day_EMA= float(request.form['30_day_EMA'])

        input_variables = pd.DataFrame([[sentinusd90momUSD,hashrate90mom,difficulty90mom,
        activeaddresses7std, difficulty7std,price14momUSD,
        sentinusdUSD, transactionvalue3stdUSD, activeaddresses3std,
        transactions3std, price30momUSD, fee_to_reward3stdUSD,
        mining_profitability90trx,sentinusd30momUSD,
        transactionvalue30momUSD, transactions, difficulty,
        difficulty14std, difficulty30mom, mining_profitability30trx,
        Change, expanding_mean, lag_1, lag_2,lag_3, lag_4,lag_5,
        lag_6, lag_7, Return, Mean, difference,d30day_WMA,
        d30_day_EMA]],columns=['sentinusd90momUSD', 'hashrate90mom', 'difficulty90mom',
       'activeaddresses7std', 'difficulty7std', 'price14momUSD',
       'sentinusdUSD', 'transactionvalue3stdUSD', 'activeaddresses3std',
       'transactions3std', 'price30momUSD', 'fee_to_reward3stdUSD',
       'mining_profitability90trx', 'sentinusd30momUSD',
       'transactionvalue30momUSD', 'transactions', 'difficulty',
       'difficulty14std', 'difficulty30mom', 'mining_profitability30trx',
       'Change', 'expanding_mean', 'lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5',
       'lag_6', 'lag_7', 'Return', 'Mean', 'difference', '30day_WMA',
       '30_day_EMA'],
                                       dtype='float',
                                       index=['input'])

        my_prediction = model.predict(input_variables)[0]
        print(my_prediction)
        return render_template('main.html',
        original_input={'sentinusd90momUSD': sentinusd90momUSD, 
        'hashrate90mom': hashrate90mom, 
        'difficulty90mom': difficulty90mom, 
        'activeaddresses7std': activeaddresses7std, 'difficulty7std': difficulty7std, 
        'price14momUSD': price14momUSD, 'sentinusdUSD': sentinusdUSD, 
        'transactionvalue3stdUSD': transactionvalue3stdUSD, 'activeaddresses3std': activeaddresses3std,
         'transactions3std': transactions3std, 'price30momUSD': price30momUSD, 
         'fee_to_reward3stdUSD': fee_to_reward3stdUSD, 'mining_profitability90trx': mining_profitability90trx,
          'sentinusd30momUSD': sentinusd30momUSD, 'transactionvalue30momUSD': transactionvalue30momUSD,
           'transactions': transactions, 'difficulty': difficulty, 'difficulty14std': difficulty14std, 
           'difficulty30mom': difficulty30mom, 'mining_profitability30trx': mining_profitability30trx, 
           'Change': Change, 'expanding_mean': expanding_mean, 'lag_1': lag_1, 'lag_2': lag_2,
            'lag_3': lag_3, 'lag_4': lag_4, 'lag_5': lag_5, 'lag_6': lag_6, 'lag_7': lag_7,
             'Return': Return, 'Mean': Mean, 'difference': difference,
              '30day_WMA': d30day_WMA, '30_day_EMA': d30_day_EMA},result=my_prediction)

if __name__ == '__main__':
	app.run(debug=True)
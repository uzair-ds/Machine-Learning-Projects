#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing liberaries
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import Normalizer, StandardScaler, MinMaxScaler, RobustScaler
import seaborn as sns
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from tabulate import tabulate
from pandas import read_csv
from numpy import mean
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
from math import sqrt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_model import ARMA
import warnings
warnings.filterwarnings('ignore', 'statsmodels.tsa.arima_model.ARMA',FutureWarning)
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# reading dataset
# This data set is collected using web scraping methods from https://bitinfocharts.c
df=pd.read_csv("DataSet.csv")
df['Date'] = pd.to_datetime(df['Date'],format='%d-%m-%Y')
df.set_index("Date", inplace = True)


# In[3]:


df.head()


# In[4]:


# checking data set contains null values
df.isnull().values.any()


# In[5]:


missed = pd.DataFrame()
missed['column'] = df.columns
missed['percent'] = [round(100* df[col].isnull().sum() / len(df), 2) for col in df.columns]
missed = missed.sort_values('percent',ascending=False)
print(missed)


# In[6]:


#!pip3 install numpy --upgrade
#!pip install numpy==1.16.5


# In[7]:


def mem_usage(pandas_obj):
    if isinstance(pandas_obj,pd.DataFrame):
        usage_b = pandas_obj.memory_usage(deep=True).sum()
    else: # we assume if not a df it's a series
        usage_b = pandas_obj.memory_usage(deep=True)
    usage_mb = usage_b / 1024 ** 2 # convert bytes to megabytes
    return "{:03.2f} MB".format(usage_mb)
print(mem_usage(df))


# In[8]:


# Visulising the price of BTC 30 day average basis
sns.set()
sns.set_style('whitegrid')
df['priceUSD'].plot(figsize=(12,6),label='price')
df['priceUSD'].rolling(window=30).mean().plot(label='30 Day Avg')# Plotting the
plt.legend()
plt.show()


# In[9]:


# ploting the no of tractions on 30 average basis
sns.set()
sns.set_style('whitegrid')
df['transactions'].plot(figsize=(12,6),label='transactions')
df['transactions'].rolling(window=30).mean().plot(label='30 Day Avg')# Plotting the
plt.legend()
plt.show()


# In[10]:


# statistcs
from tabulate import tabulate
info = [[col, df[col].count(), df[col].max(), df[col].min(),df[col].mean()] for col in df.columns]
#print(tabulate(info, headers = ['Feature', 'Count', 'Max', 'Min','Mean'], tablefmt
df1=df.reset_index(drop=True)
X=df1.drop('priceUSD', 1)
X


# In[11]:


y=df1[["priceUSD"]]


# In[12]:


y


# ## Dropping those features which is highly correlated each other

# In[13]:


# Create correlation matrix
corr_matrix = X.corr().abs()
# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
# Find index of feature columns with correlation greater than 0.75
to_drop = [column for column in upper.columns if any(upper[column] > 0.75)]


# In[14]:


# Drop features
X.drop(X[to_drop], axis=1,inplace=True)


# In[15]:


X_columns=list(X.columns)
y_columns=["priceUSD"]


# In[16]:


correlation_result={}
for i in range(len(X_columns)):
    correlation = X[X_columns[i]].corr(y["priceUSD"])
    correlation_result[X_columns[i]]=correlation
correlation_result=sorted(correlation_result.items(), key = lambda kv:(kv[1], kv[0]),reverse=True)


# In[17]:


temp=[]
for i in correlation_result:
    temp.append(i[0])
X_train2=X[temp]
X_train2


# In[18]:


top_20_features=[]
for i in range(20):
    top_20_features.append(correlation_result[i][0])


# ## Selecting Top 20 Features

# In[19]:


top_20_features
X_train=X[top_20_features]


# In[20]:


#refhttps://stackoverflow.com/questions/39409866/correlation-heatmap
# calculate the correlation matrix
corr = X_train.corr()
cmap = cmap=sns.diverging_palette(5, 250, as_cmap=True)

def magnify():
    return [dict(selector="th",
                 props=[("font-size", "7pt")]),
            dict(selector="td",
                 props=[('padding', "0em 0em")]),
            dict(selector="th:hover",
                 props=[("font-size", "12pt")]),
            dict(selector="tr:hover td:hover",
                 props=[('max-width', '200px'),
                        ('font-size', '12pt')])
]

corr.style.background_gradient(cmap, axis=1)    .set_properties(**{'max-width': '80px', 'font-size': '10pt'})    .set_caption("Hover to magify")    .set_precision(2)    .set_table_styles(magnify())


# In[21]:


X_train=df[top_20_features]
X_train.head()


# In[22]:


# Visulising the price of BTC 30 day average basis
features=list(X_train.columns)
for i in features:
    sns.set()
    sns.set_style('whitegrid')
    X_train[i].rolling(window=30).mean().plot(figsize=(12,6),label=i)
    plt.legend()
    plt.show()


# In[23]:


estimators=[]
estimators.append(['minmax',MinMaxScaler(feature_range=(-1,1))])
scale=Pipeline(estimators)
X_min_max=scale.fit_transform(X_train2)
y_min_max=scale.fit_transform(y)


# In[24]:


from sklearn.decomposition import PCA
pca = PCA(random_state=0).fit(X_min_max)
plt.plot(np.cumsum(pca.explained_variance_ratio_)*100)
plt.xlabel('number of components')
plt.ylabel('cumulative variance %')
plt.show()


# In[25]:


np.cumsum(pca.explained_variance_ratio_)


# ## Next task is to select important features
# ## Features selection using Extra tree Regressor

# In[26]:


#ref https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor
X_train, X_test, y_train, y_test = train_test_split(X_min_max, y, random_state=0)
y_train=np.ravel(y_train)
y_test=np.ravel(y_test)
clf = ExtraTreesRegressor(n_estimators=50, random_state=0).fit(X_train, y_train)
clf.score(X_test, y_test)


# In[27]:


feature_importances=[]
for i in clf.feature_importances_:
    feature_importances.append('{:f}'.format(float(i)))
    count=0
    sum=0
    index=[]
for i in range(len(feature_importances)):
    if float(feature_importances[i])>=0.001299:
    #print(i," ",feature_importances[i])
        count+=1
        sum+=float(feature_importances[i])
        index.append(i)


# In[28]:


feature_=list(X.columns)
print("List of important features are: ",[feature_[i] for i in index],"\n")
print(sum*100,"%")
print("number of important features usind tree classifier \n",count)


# ## Top 20 feature using correlation method

# In[29]:


print(top_20_features)


# In[30]:


def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))
print("common important features are")
imp_feature_=intersection(feature_, top_20_features) 
print(imp_feature_)
print("Total imp features are",len(imp_feature_))


# ## Feature engineering

# In[31]:


train_data_=df[["priceUSD",'sentinusd90momUSD', 'hashrate90mom', 'difficulty90mom', 'activeaddresses7std', 'difficulty7std', 'price14momUSD', 'sentinusdUSD', 'transactionvalue3stdUSD', 'activeaddresses3std', 'transactions3std', 'price30momUSD', 'fee_to_reward3stdUSD', 'mining_profitability90trx', 'sentinusd30momUSD', 'transactionvalue30momUSD', 'transactions', 'difficulty', 'difficulty14std', 'difficulty30mom', 'mining_profitability30trx']]


# In[32]:


train_data_


# In[33]:


train_data_.sort_index()['2010':'2022']["priceUSD"].plot(subplots=True, figsize=(15,10))
plt.savefig('bitcoin.png')
plt.show()


# ## Percent Change

# In[34]:


#train_data_['Change'] = train_data_.priceUSD.div(train_data_.priceUSD.shift())
train_data_ = train_data_.assign(Change=pd.Series(train_data_.priceUSD.div(train_data_.priceUSD.shift())))
train_data_['Change'].plot(figsize=(20,8))
plt.show()


# ## Expanding Mean

# In[35]:


#train_data_['expanding_mean'] = train_data_['priceUSD'].expanding(1).mean()
train_data_ = train_data_.assign(expanding_mean=pd.Series(train_data_['priceUSD'].expanding(1).mean()))
train_data_['expanding_mean'].plot(figsize=(20,8))
plt.show()


# ## Lag Feature

# In[36]:


train_data_['lag_1'] = train_data_['priceUSD'].shift(1)
train_data_['lag_2'] = train_data_['priceUSD'].shift(2)
train_data_['lag_3'] = train_data_['priceUSD'].shift(3)
train_data_['lag_4'] = train_data_['priceUSD'].shift(4)
train_data_['lag_5'] = train_data_['priceUSD'].shift(5)
train_data_['lag_6'] = train_data_['priceUSD'].shift(6)
train_data_['lag_7'] = train_data_['priceUSD'].shift(7)


# ## Return

# In[37]:


train_data_ = train_data_.assign(Return=pd.Series(train_data_.Change.sub(1).mul(100)))
train_data_['Return'].plot(figsize=(20,8))
plt.show()


# In[38]:


train_data_.priceUSD.pct_change().mul(100).plot(figsize=(20,6))
plt.show()


# ## Window Functions

# In[39]:


train_data_ = train_data_.assign(Mean=pd.Series(train_data_['priceUSD'].rolling(window=30).mean()))
train_data_['Mean'].plot(figsize=(20,8),label='mean price')
train_data_['priceUSD'].plot(label='original')
plt.legend()
plt.show()


# ## Time series decomposition and Random walks

# In[40]:


train_data_["priceUSD"].plot(figsize=(25,10))
plt.show()


# In[41]:


# Now, for decomposition...
from pylab import rcParams
import statsmodels.api as sm
rcParams['figure.figsize'] = 25, 15
decomposed_train_data_ = sm.tsa.seasonal_decompose(train_data_["priceUSD"],period=365) # The frequncy is annual
figure = decomposed_train_data_.plot()
plt.show()


# In[42]:


train_data_.isnull().values.any()


# In[43]:


train_data_.dropna(axis = 0, how ='any',inplace=True)
train_data_.isnull().values.any()


# In[44]:


train_data_["priceUSD"].describe()


# In[45]:



from scipy import signal
detrended = signal.detrend(train_data_["priceUSD"].values)
plt.plot(detrended)
plt.title('detrended by subtracting the least squares fit', fontsize=16)


# In[46]:


# Plotting white noise
from random import gauss
from random import seed
from pandas import Series
from pandas.plotting import autocorrelation_plot
series = Series(train_data_["priceUSD"])
# summary stats
print(train_data_["priceUSD"].describe())


# In[47]:


# histogram plot
series.hist()
plt.show()


# In[48]:


# autocorrelation
autocorrelation_plot(series)
plt.show()


# In[49]:


from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plot_pacf(train_data_["priceUSD"],lags=20)
plt.show()


# In[50]:


# Plotting autocorrelation of white noise
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plot_acf(train_data_["priceUSD"],lags=150,alpha=0.05)
plt.show()


# 
# "The partial autocorrelation function shows a high correlation with the first lag and lesser
# correlation with the second and third lag. The autocorrelation function shows a slow decay,
# which means that the future values have a very high correlation with its past values.
# As we can see, the time series contains significant auto-correlations up through lags 130"

# In[51]:


import statsmodels.stats.diagnostic as diag
diag.acorr_ljungbox(train_data_["priceUSD"], lags=[140], boxpierce=True)


# "
# The value 256068.08656797 is the test statistic of the Box-Pierce test and 0.0 is its p-value as per
# the Chi-square(k=140) tables.
# As we can see, both p-values are less than 0.01 and so we can say
# with 99% confidence that the time series is not pure white noise."

# ## Random Walk

# In[52]:


# Augmented Dickey-Fuller test on volume of google and microsoft stocks
#https://www.statsmodels.org/dev/_modules/statsmodels/tsa/stattools.html
from statsmodels.tsa.stattools import adfuller
adf = adfuller(train_data_["priceUSD"])
print("p-value : {}".format(float(adf[1])))


# ## Generating a Random Walk

# In[53]:


diff_Y_i = train_data_["priceUSD"].diff()
train_data_ = train_data_.assign(difference=pd.Series(diff_Y_i))
#drop the NAN in the first row
diff_Y_i = diff_Y_i.dropna()
diff_Y_i.plot()
plt.show()


# In[54]:


import statsmodels.stats.diagnostic as diag
diag.acorr_ljungbox(diff_Y_i, lags=[140], boxpierce=True)


# ## Stationarity

# In[55]:


# non stationary
decomposed_train_data_.trend.plot()
plt.show()


# In[56]:


# The new stationary plot
decomposed_train_data_.trend.diff().plot()
plt.show()


# In[57]:


train_data_


# ## Modelling using statstools autoregressive (AR) mode Forecasting a simulated model

# In[58]:


print(train_data_.isnull().values.sum())
train_data_.dropna(axis = 0, how ='any',inplace=True)
print(train_data_.isnull().values.sum())


# In[59]:


# prepare situation
def moving_average_(data):
    X = data
    window = 3
    history = [X[i] for i in range(window)]
    test = [X[i] for i in range(window, len(X))]
    predictions = list()
    # walk forward over time steps in test
    for t in range(len(test)):
        length = len(history)
        yhat = mean([history[i] for i in range(length-window,length)])
        obs = test[t]
        predictions.append(yhat)
        history.append(obs)
        #print('predicted=%f, expected=%f' % (yhat, obs))
    error = mean_squared_error(test, predictions)
    print('Test MSE: %.3f' % error)
    # zoom plot
    pyplot.plot(test[0:100],label="Original")
    pyplot.plot(predictions[0:100], color='red',label="Prediction")
    plt.legend()
    pyplot.show()


# In[60]:


moving_average_(train_data_["priceUSD"].values)


# In[61]:


df_train = train_data_[train_data_.index < "2019"]
df_valid = train_data_[train_data_.index >= "2019"]


# In[62]:


def exponential_moving_():
    weights = np.arange(1,31) #this creates an array with integers 1 to 31 included
    weights
    wma10 = train_data_["priceUSD"].rolling(30).apply(lambda prices: np.dot(prices, weights)/weights.sum(), raw=True)
    train_data_['30day_WMA'] = np.round(wma10, decimals=3)
    #sma10 = train_data_['priceUSD'].rolling(30).mean()
    temp = train_data_.dropna(how='any',axis=0) 
    print(sqrt(mean_squared_error(temp.priceUSD, temp['30day_WMA'])))
    plt.figure(figsize = (12,6))
    plt.plot(train_data_['priceUSD'], label="Price")
    plt.plot(wma10, label="30-Day WMA")
    #plt.plot(sma10, label="10-Day SMA")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.show()


# In[63]:


exponential_moving_()


# ## Exponential Moving Average

# In[64]:


def exponential_moving_average():
    ema30 = train_data_['priceUSD'].ewm(span=30).mean()
    train_data_['30_day_EMA'] = np.round(ema30, decimals=3)
    print(sqrt(mean_squared_error(train_data_.priceUSD, train_data_['30_day_EMA'])))
    plt.figure(figsize = (12,6))
    plt.plot(train_data_['priceUSD'], label="Price")
    plt.plot(ema30, label="30-Day WMA")
    #plt.plot(sma10, label="10-Day SMA")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.show()


# In[65]:


exponential_moving_average()


# In[66]:


# Predicting closing prices 
humid = ARMA(train_data_["priceUSD"].diff().iloc[1:].values, order=(10,0))
res = humid.fit()
res.plot_predict(start=3500, end=3800)
plt.show();


# In[67]:


'''
from pandas import read_csv
from numpy import mean
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
# prepare situation
X = train_data_["priceUSD"].values
window = 3
history = [X[i] for i in range(window)]
test = [X[i] for i in range(window, len(X))]
predictions = list()
# walk forward over time steps in test
for t in range(len(test)):
    length = len(history)
    yhat = mean([history[i] for i in range(length-window,length)])
    obs = test[t]
    predictions.append(yhat)
    history.append(obs)
    print('predicted=%f, expected=%f' % (yhat, obs))
    error = mean_squared_error(test, predictions)
    print('Test MSE: %.3f' % error)
    # zoom plot
pyplot.plot(test[0:100],label="Original")
pyplot.plot(predictions[0:100], color='red',label="Prediction")
plt.legend()
pyplot.show()'''


# ## Prediction using ARIMA model

# In[68]:


#train_data_.columns
train_data_.fillna(method='ffill', inplace=True)
train_data_.fillna(method='backfill', inplace=True)


# In[69]:


train_data_.isnull().values.any()


# In[70]:


#train_data_.columns


# In[71]:


df_train = train_data_[train_data_.index < "2019"]
df_valid = train_data_[train_data_.index >= "2019"]


# In[72]:


important_feature_=['sentinusd90momUSD','hashrate90mom','difficulty90mom',
                    'activeaddresses7std', 'difficulty7std','price14momUSD',
                    'sentinusdUSD', 'transactionvalue3stdUSD','activeaddresses3std',
                    'transactions3std', 'price30momUSD','fee_to_reward3stdUSD',
                    'mining_profitability90trx','sentinusd30momUSD',
                    'transactionvalue30momUSD', 'transactions','difficulty',
                    'difficulty14std', 'difficulty30mom', 'mining_profitability30trx',
                    'Change','expanding_mean','lag_1','lag_2','lag_3','lag_4','lag_5',
                    'lag_6','lag_7','Return','Mean','difference','30day_WMA','30_day_EMA']


# In[73]:


get_ipython().system('pip install pmdarima')
from pmdarima import auto_arima
model = auto_arima(df_train.priceUSD, exogenous=df_train[important_feature_], trace=True, error_action="ignore", suppress_warnings=True)
model.fit(df_train.priceUSD, exogenous=df_train[important_feature_])


# In[74]:


forecast = model.predict(n_periods=len(df_valid), exogenous=df_valid[important_feature_])
df_valid = df_valid.assign(Forecast_ARIMAX=forecast)
#df_valid["Forecast_ARIMAX"] = forecast


# In[75]:


df_valid[["priceUSD", "Forecast_ARIMAX"]].plot(figsize=(14, 7))
plt.show()


# In[76]:


print("RMSE of Auto ARIMAX:", np.sqrt(mean_squared_error(df_valid.priceUSD, df_valid.Forecast_ARIMAX)))
print("\nMAE of Auto ARIMAX:", mean_absolute_error(df_valid.priceUSD, df_valid.Forecast_ARIMAX))


# ## Stacking Classifier

# In[106]:


from sklearn.model_selection import train_test_split


# In[113]:


train_data_.columns.value_counts().sum()


# In[115]:


X = train_data_.drop(columns='priceUSD')
y = train_data_['priceUSD']


# In[116]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)


# ## Model 1 Random Forest

# In[82]:


from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
rf = RandomForestRegressor()
# Randomized Search CV
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 10, stop = 1000, num = 12)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
# max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 15, 100]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 5, 10]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
    'max_features': max_features,
    'max_depth': max_depth,
    'min_samples_split': min_samples_split}
print(random_grid)


# In[ ]:





# In[117]:


from sklearn.model_selection import RandomizedSearchCV
# Use the random grid to search for best hyperparameters
# First create the base model to tune
random_grid = {'n_estimators': [5,10,20,50,100,200, 400, 600],
                'max_features': ['auto', 'sqrt'],
                'max_depth': [5,10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
                'min_samples_split': [2, 5, 10,20],
                'min_samples_leaf': [1, 2, 4,6,8,10],
                 'bootstrap': [True, False]}

rf = RandomForestRegressor()
# Random search of parameters, using 5 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 50, cv = 5, verbose=2, random_state=1000, n_jobs = -1)
# Fit the random search model
#X = X.values.astype(np.float)
#y = y.values.astype(np.float)
rf_random.fit(X_train, y)


# In[118]:


rf_random.best_params_


# ## Final Model

# ### Checking if there is any large difference in  predicted values and the real true values

# In[126]:


print(rf_random.predict(X_train.tail())) # predicted 
y_train.tail() #true values 


# In[137]:


y_pred = rf_random.predict(X_test)


# ### r_square value calculation

# In[129]:


from sklearn import metrics


# In[133]:


X_test.columns


# In[142]:


r_square = metrics.r2_score(y_test, y_pred)
print(round(r_square,5))


# In[ ]:


print(round(r_square,5))


# ### Mean Absolute Error and Mean Squared Error Calculation

# In[144]:


from sklearn import metrics
mae = round(metrics.mean_absolute_error(y_test, y_pred), 5)
mse= round(metrics.mean_squared_error(y_test, y_pred), 5)


# In[145]:


print(mae,"\n",mse)


# ## Saving a Model 

# In[146]:


import pickle
#open a file, where you ant to store the data
file = open('rf_random.pkl', 'wb')

#dump information to that file
pickle.dump(rf_random, file)


# In[ ]:





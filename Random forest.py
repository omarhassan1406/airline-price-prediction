import pandas as pd
from sklearn import preprocessing
import ast
from pandas import Timedelta
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import seaborn as sns
from sklearn.metrics import r2_score
from sklearn.ensemble import  RandomForestRegressor
#preprocessing
def encoder(df,col):
    lbl = preprocessing.LabelEncoder()
    lbl.fit(df[col])
    df[col] = lbl.transform(df[col])

def featureScaling(df , col):
    max_a = df[col].max()
    min_a = df[col].min()
    min_norm = -1
    max_norm =1
    df[col] = (df[col]- min_a) *(max_norm - min_norm) / (max_a-min_a) + min_norm


df = pd.read_csv("airline-price-prediction.csv")

encoder(df,'airline')
encoder(df,'ch_code')

df['arr'] = df['date'] + " " + df['arr_time']
df['dep'] = df['date'] + " " + df['dep_time']
df = df.drop('arr_time', 1)
df = df.drop('dep_time', 1)
df['date'] = pd.to_datetime(df['date'])
df['arr'] = pd.to_datetime(df['arr'])
df['dep'] = pd.to_datetime(df['dep'])

df['time_taken'].replace('h m','',regex=True,inplace=True)
df['time_taken'].replace('h ','.',regex=True,inplace=True)
df['time_taken'].replace('m','',regex=True,inplace=True)
df['time_taken'] = pd.to_numeric(df['time_taken'])



df['stop'].replace('\n\t*','',regex=True,inplace=True)
encoder(df,'stop')

dum = pd.get_dummies(df.type, prefix='type')
df['type_business'] = dum['type_business']
df['type_economy'] = dum['type_economy']
df = df.drop('type', 1)

x = df["route"].apply(lambda x: ast.literal_eval(x))
sor = []
for i in range(len(df['route'])):
    sor.append(x[i]['source'])
df['source'] = sor
dist = []
for i in range(len(df['route'])):
    dist.append(x[i]['destination'])
df['destination'] = dist
df = df.drop('route', 1)

encoder(df,'source')
encoder(df,'destination')

df['price'].replace(',..,','.',regex=True,inplace=True)
df['price'].replace(',','',regex=True,inplace=True)
df['price'] = pd.to_numeric(df['price'])




df = df[['date','airline','ch_code','num_code','dep','time_taken','stop','arr','type_business','type_economy','source','destination','price']]
for i in range(0,12):
    featureScaling(df , df.columns[i])
#modeling

X=df.iloc[:,0:12]  #Features
Y=df['price']  #Label
care_data = df.iloc[:,:]

#Corellation
correlation = care_data.corr()
top_features = correlation.index[abs(correlation['price']) > 0]

#Correlation plot
plt.subplots(figsize=(12, 8))
top_corr = care_data[top_features].corr()
sns.heatmap(top_corr, annot=True)
plt.show()
top_feature = top_features.delete(-1)
X = X[top_feature]


##poly_features = PolynomialFeatures(degree=4)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.20, random_state=4)

regressor = RandomForestRegressor(n_estimators=100 , random_state=0)
regressor.fit(X_train , y_train)

prediction = regressor.predict(X_test)
##X_train_poly = poly_features.fit_transform(X_train)
# fit the transformed features to Linear Regression
##poly_model = linear_model.LinearRegression()
##poly_model.fit(X_train_poly , y_train)

# predicting on training data-set
##y_train_predicted = poly_model.predict(X_train_poly)

# predicting on test data-set
##prediction = poly_model.predict(poly_features.fit_transform(X_test))


print('Mean Square Error = ', metrics.mean_squared_error(y_test, prediction))
true_value = np.asarray(y_test)[0]
predicted_value =prediction[0]
print('True value for the first in the test set : ' + str(true_value))
print('Predicted value for the first in the test set  : ' + str(predicted_value))
print('SCORE or accuracy of model = ' + str(r2_score(y_test,prediction)))


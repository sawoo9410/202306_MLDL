# -*- coding: utf-8 -*-
"""
Anomaly Detection for Time Series Sensor Data (pump_sensor_data) 

https://www.kaggle.com/code/pinakimishrads/anomaly-detection-for-time-series-sensor-data
Created on Sat Jun  3 15:52:37 2023
"""

##--- Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')
sns.set(style = "whitegrid",font_scale = 1.5)
%matplotlib inline
plt.rcParams['figure.figsize']=[12,8]


##--- Importing and exploring Dataset
sensor_df = pd.read_csv('sensor.csv')

# Basic Exploration
print("The dataset has " , sensor_df.shape[0],"rows and", sensor_df.shape[1], "columns")
#The dataset has  220320 rows and 55 columns

# First and last 5 rows
sensor_df
"""
        Unnamed: 0            timestamp  ...  sensor_51  machine_status
0                0  2018-04-01 00:00:00  ...   201.3889          NORMAL
1                1  2018-04-01 00:01:00  ...   201.3889          NORMAL
2                2  2018-04-01 00:02:00  ...   203.7037          NORMAL
3                3  2018-04-01 00:03:00  ...   203.1250          NORMAL
4                4  2018-04-01 00:04:00  ...   201.3889          NORMAL
           ...                  ...  ...        ...             ...
220315      220315  2018-08-31 23:55:00  ...   231.1921          NORMAL
220316      220316  2018-08-31 23:56:00  ...   231.1921          NORMAL
220317      220317  2018-08-31 23:57:00  ...   232.0602          NORMAL
220318      220318  2018-08-31 23:58:00  ...   234.0856          NORMAL
220319      220319  2018-08-31 23:59:00  ...   234.0856          NORMAL

[220320 rows x 55 columns]
"""

# Basic information
sensor_df.info()
"""
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 220320 entries, 0 to 220319
Data columns (total 55 columns):
 #   Column          Non-Null Count   Dtype  
---  ------          --------------   -----  
 0   Unnamed: 0      220320 non-null  int64  
 1   timestamp       220320 non-null  object 
 2   sensor_00       210112 non-null  float64
 3   sensor_01       219951 non-null  float64
 4   sensor_02       220301 non-null  float64
 5   sensor_03       220301 non-null  float64
 6   sensor_04       220301 non-null  float64
 7   sensor_05       220301 non-null  float64
 8   sensor_06       215522 non-null  float64
 9   sensor_07       214869 non-null  float64
 10  sensor_08       215213 non-null  float64
 11  sensor_09       215725 non-null  float64
 12  sensor_10       220301 non-null  float64
 13  sensor_11       220301 non-null  float64
 14  sensor_12       220301 non-null  float64
 15  sensor_13       220301 non-null  float64
 16  sensor_14       220299 non-null  float64
 17  sensor_15       0 non-null       float64
 18  sensor_16       220289 non-null  float64
 19  sensor_17       220274 non-null  float64
 20  sensor_18       220274 non-null  float64
 21  sensor_19       220304 non-null  float64
 22  sensor_20       220304 non-null  float64
 23  sensor_21       220304 non-null  float64
 24  sensor_22       220279 non-null  float64
 25  sensor_23       220304 non-null  float64
 26  sensor_24       220304 non-null  float64
 27  sensor_25       220284 non-null  float64
 28  sensor_26       220300 non-null  float64
 29  sensor_27       220304 non-null  float64
 30  sensor_28       220304 non-null  float64
 31  sensor_29       220248 non-null  float64
 32  sensor_30       220059 non-null  float64
 33  sensor_31       220304 non-null  float64
 34  sensor_32       220252 non-null  float64
 35  sensor_33       220304 non-null  float64
 36  sensor_34       220304 non-null  float64
 37  sensor_35       220304 non-null  float64
 38  sensor_36       220304 non-null  float64
 39  sensor_37       220304 non-null  float64
 40  sensor_38       220293 non-null  float64
 41  sensor_39       220293 non-null  float64
 42  sensor_40       220293 non-null  float64
 43  sensor_41       220293 non-null  float64
 44  sensor_42       220293 non-null  float64
 45  sensor_43       220293 non-null  float64
 46  sensor_44       220293 non-null  float64
 47  sensor_45       220293 non-null  float64
 48  sensor_46       220293 non-null  float64
 49  sensor_47       220293 non-null  float64
 50  sensor_48       220293 non-null  float64
 51  sensor_49       220293 non-null  float64
 52  sensor_50       143303 non-null  float64
 53  sensor_51       204937 non-null  float64
 54  machine_status  220320 non-null  object 
dtypes: float64(52), int64(1), object(2)
memory usage: 92.5+ MB
"""


##--- Ddata Cleaning 

#no values for sensor 15 and unnamed column is unnecessary
sensor_df.drop(['sensor_15','Unnamed: 0'],inplace = True,axis=1)
sensor_df.head()
"""
             timestamp  sensor_00  ...  sensor_51  machine_status
0  2018-04-01 00:00:00   2.465394  ...   201.3889          NORMAL
1  2018-04-01 00:01:00   2.465394  ...   201.3889          NORMAL
2  2018-04-01 00:02:00   2.444734  ...   203.7037          NORMAL
3  2018-04-01 00:03:00   2.460474  ...   203.1250          NORMAL
4  2018-04-01 00:04:00   2.445718  ...   201.3889          NORMAL

[5 rows x 53 columns]
"""

#check percentage of missing values for each column
(sensor_df.isnull().sum().sort_values(ascending=False)/len(sensor_df))*100
"""
sensor_50         34.956881
sensor_51          6.982117
sensor_00          4.633261
sensor_07          2.474129
sensor_08          2.317992
sensor_06          2.177741
sensor_09          2.085603
sensor_01          0.167484
sensor_30          0.118464
sensor_29          0.032680
sensor_32          0.030864
sensor_17          0.020879
sensor_18          0.020879
sensor_22          0.018609
sensor_25          0.016340
sensor_16          0.014070
sensor_49          0.012255
sensor_48          0.012255
sensor_47          0.012255
sensor_46          0.012255
sensor_45          0.012255
sensor_44          0.012255
sensor_43          0.012255
sensor_42          0.012255
sensor_41          0.012255
sensor_40          0.012255
sensor_39          0.012255
sensor_38          0.012255
sensor_14          0.009532
sensor_26          0.009078
sensor_03          0.008624
sensor_10          0.008624
sensor_13          0.008624
sensor_12          0.008624
sensor_11          0.008624
sensor_05          0.008624
sensor_04          0.008624
sensor_02          0.008624
sensor_36          0.007262
sensor_37          0.007262
sensor_28          0.007262
sensor_27          0.007262
sensor_31          0.007262
sensor_35          0.007262
sensor_24          0.007262
sensor_23          0.007262
sensor_34          0.007262
sensor_21          0.007262
sensor_20          0.007262
sensor_19          0.007262
sensor_33          0.007262
timestamp          0.000000
machine_status     0.000000
dtype: float64
"""

#too many missing values in sensor 50 , so dropping that
sensor_df.drop('sensor_50',inplace = True,axis=1)
sensor_df.head()
"""
             timestamp  sensor_00  ...  sensor_51  machine_status
0  2018-04-01 00:00:00   2.465394  ...   201.3889          NORMAL
1  2018-04-01 00:01:00   2.465394  ...   201.3889          NORMAL
2  2018-04-01 00:02:00   2.444734  ...   203.7037          NORMAL
3  2018-04-01 00:03:00   2.460474  ...   203.1250          NORMAL
4  2018-04-01 00:04:00   2.445718  ...   201.3889          NORMAL

[5 rows x 52 columns]
"""

# convert time to index
sensor_df['index'] = pd.to_datetime(sensor_df['timestamp'])
sensor_df.index = sensor_df['index']

#Drop index and timestamp columns
sensor_df.drop(['index','timestamp'],inplace = True,axis=1)
sensor_df.head()
"""
                     sensor_00  sensor_01  ...  sensor_51  machine_status
index                                      ...                           
2018-04-01 00:00:00   2.465394   47.09201  ...   201.3889          NORMAL
2018-04-01 00:01:00   2.465394   47.09201  ...   201.3889          NORMAL
2018-04-01 00:02:00   2.444734   47.35243  ...   203.7037          NORMAL
2018-04-01 00:03:00   2.460474   47.09201  ...   203.1250          NORMAL
2018-04-01 00:04:00   2.445718   47.13541  ...   201.3889          NORMAL

[5 rows x 51 columns]
"""


##--- Dealing with missing values

# Imputing missing values with mean
sensor_df.fillna(sensor_df.mean(),inplace= True)
sensor_df.isnull().sum()
"""
sensor_00         0
sensor_01         0
sensor_02         0
sensor_03         0
sensor_04         0
sensor_05         0
sensor_06         0
sensor_07         0
sensor_08         0
sensor_09         0
sensor_10         0
sensor_11         0
sensor_12         0
sensor_13         0
sensor_14         0
sensor_16         0
sensor_17         0
sensor_18         0
sensor_19         0
sensor_20         0
sensor_21         0
sensor_22         0
sensor_23         0
sensor_24         0
sensor_25         0
sensor_26         0
sensor_27         0
sensor_28         0
sensor_29         0
sensor_30         0
sensor_31         0
sensor_32         0
sensor_33         0
sensor_34         0
sensor_35         0
sensor_36         0
sensor_37         0
sensor_38         0
sensor_39         0
sensor_40         0
sensor_41         0
sensor_42         0
sensor_43         0
sensor_44         0
sensor_45         0
sensor_46         0
sensor_47         0
sensor_48         0
sensor_49         0
sensor_51         0
machine_status    0
dtype: int64
"""


##--- EDA

sensor_df.machine_status.value_counts()
"""
NORMAL        205836
RECOVERING     14477
BROKEN             7
Name: machine_status, dtype: int64
"""

# Machine status -  pie chart
stroke_labels = ["Normal","Recovering","Broken"]
sizes = sensor_df.machine_status.value_counts()

plt.pie(x=sizes,labels=stroke_labels)  
plt.show()

# Extract the readings from the Broken state of the pump
broken= sensor_df[sensor_df['machine_status']=='BROKEN']

# Extract the name of the numerical columns
sensor_df_2 = sensor_df.drop(['machine_status'],axis=1)
names= sensor_df_2.columns

# Plot timeseries for each sensor with Broken state marked with X in red color

for name in names:
    _=plt.figure(figsize=(18,3))
    _=plt.plot(broken[name],linestyle='none',marker='X',color='red',markersize=12)
    _=plt.plot(sensor_df[name],color='blue')
    _=plt.title(name)
    plt.show()


##--- Data Preprocessing and Dimensionality Reduction

## (1) Scaling the data
from sklearn.preprocessing import StandardScaler

#dropping the target column from the dataframe
sensor_df_2 = sensor_df.drop('machine_status',axis=1)
col_names=sensor_df_2.columns

#scaling
scaler=StandardScaler()
sensor_df_2_scaled= scaler.fit_transform(sensor_df_2)
sensor_df_2_scaled = pd.DataFrame(sensor_df_2_scaled,columns=col_names)

sensor_df_2_scaled.head()
"""
   sensor_00  sensor_01  sensor_02  ...  sensor_48  sensor_49  sensor_51
0   0.231450  -0.151675   0.639386  ...   0.086297   0.553138  -0.012402
1   0.231450  -0.151675   0.639386  ...   0.086297   0.553138  -0.012402
2   0.180129  -0.072613   0.639386  ...   0.061668   0.522906   0.009499
3   0.219228  -0.151675   0.627550  ...   0.061668   0.507790   0.004024
4   0.182573  -0.138499   0.639386  ...   0.089816   0.492674  -0.012402

[5 rows x 50 columns]
"""

## (2) Principal Component Analysis
from sklearn.decomposition import PCA

pca = PCA()
pca.fit(sensor_df_2_scaled)
#Out: PCA()

# Plot the principal components
features = range(pca.n_components_)
_=plt.figure(figsize=(22,5))
_=plt.bar(features,pca.explained_variance_)
_=plt.xlabel('PCA feature')
_=plt.ylabel('Variance')
_=plt.xticks(features)
_=plt.title('Important Principal Component')
plt.show()

#calculate PCA with 2 components
pca = PCA(n_components=2)
pComponents = pca.fit_transform(sensor_df_2_scaled)
principal_df = pd.DataFrame(data = pComponents,columns=['pca1','pca2'])

principal_df.head()
"""
       pca1      pca2
0 -0.046056  0.490524
1 -0.046056  0.490524
2 -0.186309  0.500354
3 -0.186651  0.538034
4 -0.142655  0.645878
"""

## (3) Stationarity & Autocorrelation
## Stationarity <= 시계열 데이터가 Stationarity 인지를 검정
from statsmodels.tsa.stattools import adfuller

# Run Augmented Dickey Fuller Test
result = adfuller(principal_df['pca1'])

# Print p-value 
print(result[1])
#0.00014222614547431883

## Autocorrelation

# Compute change in daily mean
pca1= principal_df['pca1'].pct_change()

# Compute Autocorelation
autoco = pca1.dropna().autocorr()
print('Autocorrelation is:',autoco)
#Autocorrelation is: -7.2161716299099175e-06

# Plot ACF
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(pca1.dropna(),lags=20,alpha=0.05)
plt.show()


##--- Modeling

## (1) Model-1 K-Means Clusting

# Import kmeans
from sklearn.cluster import KMeans

# Initialize and fit kmeans
#!pip install --upgrade threadpoolctl <= 이 코드를 실행해야할지도 모름
kmeans = KMeans(n_clusters=2,random_state=13)
kmeans.fit(principal_df.values)

KMeans(n_clusters=2, random_state=13)

# Prediction
labels = kmeans.predict(principal_df.values)

# Plotting the clusters
_=plt.figure(figsize=(9,7))
_=plt.scatter(principal_df['pca1'],principal_df['pca2'],c=labels)
_=plt.xlabel('pca1')
_=plt.ylabel('pca2')
_=plt.title('K-Means of clusterings')
plt.show()

# Write a function that calucalates distance between each point and the centroid of the closest cluster
def getDistanceByPoint(data, model):
    """ Function that calculates the distance between a point and centroid of a cluster, 
            returns the distances in pandas series"""
    distance = []
    for i in range(0,len(data)):
        Xa = np.array(data.loc[i])
        Xb = model.cluster_centers_[model.labels_[i]-1]
        distance.append(np.linalg.norm(Xa-Xb))
    return pd.Series(distance, index=data.index)

# Assume that 13% of the entire data set are anomalies 
outliers_fraction = 0.13

# Get the distance between each point and its nearest centroid. 
# the biggest distances are considered as anomaly
distance = getDistanceByPoint(principal_df, kmeans)

# Number of observations that equate to the 13% of the entire data set
number_of_outliers = int(outliers_fraction*len(distance))

# Take the minimum of the largest 13% of the distances as the threshold
threshold = distance.nlargest(number_of_outliers).min()

# Anomaly1 contains the anomaly result of the above method Cluster (0:normal, 1:anomaly) 
principal_df['kmeans_anomaly'] = (distance >= threshold).astype(int)

principal_df.head()
"""
       pca1      pca2  kmeans_anomaly
0 -0.046056  0.490524               0
1 -0.046056  0.490524               0
2 -0.186309  0.500354               0
3 -0.186651  0.538034               0
4 -0.142655  0.645878               0
"""	

principal_df["kmeans_anomaly"].value_counts()
"""
0    191679
1     28641
Name: kmeans_anomaly, dtype: int64
"""

## (2) Visualization over different sensors
dfBroken = sensor_df[sensor_df["machine_status"]=="BROKEN"]

sensor_df['kmeans_anomaly'] = pd.Series(principal_df['kmeans_anomaly'].values, index=sensor_df.index)
a = sensor_df[sensor_df['kmeans_anomaly'] == 1] #anomaly
_ = plt.figure(figsize=(18,6))
_ = plt.plot(sensor_df['sensor_00'], color='blue', label='Normal')
_ = plt.plot(a['sensor_00'], linestyle='none', marker='X', color='red', markersize=12, label='KMeans Anomaly')
_ = plt.plot(dfBroken['sensor_00'], linestyle='none', marker='X', color='green', markersize=12, label='Broken')
_ = plt.xlabel('Date and Time')
_ = plt.ylabel('Sensor Reading')
_ = plt.title('Sensor_00 Anomalies')
_ = plt.legend(loc='best')
plt.show()

sensor_df['kmeans_anomaly'] = pd.Series(principal_df['kmeans_anomaly'].values, index=sensor_df.index)
a = sensor_df[sensor_df['kmeans_anomaly'] == 1] #anomaly
_ = plt.figure(figsize=(18,6))
_ = plt.plot(sensor_df['sensor_11'], color='blue', label='Normal')
_ = plt.plot(a['sensor_11'], linestyle='none', marker='X', color='red', markersize=12, label='KMeans Anomaly')
_ = plt.plot(dfBroken['sensor_11'], linestyle='none', marker='X', color='green', markersize=12, label='Broken')
_ = plt.xlabel('Date and Time')
_ = plt.ylabel('Sensor Reading')
_ = plt.title('Sensor_11 Anomalies')
_ = plt.legend(loc='best')
plt.show();

## (3) Model 2 - Isolation Forest
# Import IsolationForest
from sklearn.ensemble import IsolationForest

# Fit and predict
model_if =  IsolationForest(random_state=13)

model_if.fit(principal_df.drop('kmeans_anomaly', axis = 1)) 

principal_df['if_anomaly'] = pd.Series(model_if.predict(principal_df.drop('kmeans_anomaly', axis = 1)))

principal_df['if_anomaly'].value_counts()
"""
 1    172555
-1     47765
Name: if_anomaly, dtype: int64
"""

sensor_df['if_anomaly'] = pd.Series(principal_df['if_anomaly'].values, index=sensor_df.index)
a = sensor_df[sensor_df['if_anomaly'] == -1] #anomaly
_ = plt.figure(figsize=(18,6))
_ = plt.plot(sensor_df['sensor_00'], color='blue', label='Normal')
_ = plt.plot(a['sensor_00'], linestyle='none', marker='X', color='red', markersize=12, label='Anomaly')
_ = plt.plot(dfBroken['sensor_00'], linestyle='none', marker='X', color='green', markersize=12, label='Broken')
_ = plt.xlabel('Date and Time')
_ = plt.ylabel('Sensor Reading')
_ = plt.title('Sensor_00 Anomalies')
_ = plt.legend(loc='best')
plt.show();

sensor_df['if_anomaly'] = pd.Series(principal_df['if_anomaly'].values, index=sensor_df.index)
a = sensor_df[sensor_df['if_anomaly'] == -1] #anomaly
_ = plt.figure(figsize=(18,6))
_ = plt.plot(sensor_df['sensor_11'], color='blue', label='Normal')
_ = plt.plot(a['sensor_11'], linestyle='none', marker='X', color='red', markersize=12, label='Anomaly')
_ = plt.plot(dfBroken['sensor_00'], linestyle='none', marker='X', color='green', markersize=12, label='Broken')
_ = plt.xlabel('Date and Time')
_ = plt.ylabel('Sensor Reading')
_ = plt.title('Sensor_11 Anomalies')
_ = plt.legend(loc='best')
plt.show();

## (4) Evaluation
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, RocCurveDisplay

predictions = sensor_df[['machine_status', 'kmeans_anomaly', 'if_anomaly']]

# if anomaly uniformity. 1 is 0 and -1 is 1
predictions.loc[predictions["if_anomaly"] == 1, "if_anomaly"] = 0
predictions.loc[predictions["if_anomaly"] == -1, "if_anomaly"] = 1

# turning machine status numerical
predictions["machine_status"] = predictions["machine_status"].map(
    {"NORMAL": 0, "RECOVERING": 1, "BROKEN": 1})

## (5) K-Means Evaluation
# classification report
print(classification_report(predictions['machine_status'].values, 
                            predictions['kmeans_anomaly'].values))
"""
              precision    recall  f1-score   support

           0       0.99      0.93      0.96    205836
           1       0.47      0.93      0.62     14484

    accuracy                           0.93    220320
   macro avg       0.73      0.93      0.79    220320
weighted avg       0.96      0.93      0.94    220320
"""

# Confusion matrix
ConfusionMatrixDisplay.from_predictions(predictions['machine_status'].values,
                                        predictions['kmeans_anomaly'].values)
plt.show()


# ROC curve
RocCurveDisplay.from_predictions(predictions['machine_status'], 
                                 predictions['kmeans_anomaly'])
plt.show()

## (6) Isolation Forest Evaluation
print(classification_report(predictions['machine_status'], 
                            predictions['if_anomaly']))
"""
              precision    recall  f1-score   support

           0       1.00      0.84      0.91    205836
           1       0.30      0.98      0.46     14484

    accuracy                           0.85    220320
   macro avg       0.65      0.91      0.68    220320
weighted avg       0.95      0.85      0.88    220320
"""

ConfusionMatrixDisplay.from_predictions(predictions['machine_status'], 
                                        predictions['if_anomaly'])
plt.show()

RocCurveDisplay.from_predictions(predictions['machine_status'], 
                                 predictions['if_anomaly'])
plt.show()







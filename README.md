# DEPI
DEPI Preprocessing
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


data = pd.read_csv(r'D:\csvs\nnn\worldometer_data.csv')

print(data)

print(data.isnull().sum())
print(data.duplicated().sum())
print(data.dtypes)
data = data.dropna()
data = data.drop_duplicates()


from sklearn import preprocessing
pr_data = preprocessing.LabelEncoder()
dtype = data.dtypes
for i in range(data.shape[1]):
    if dtype[i] == 'object':
        modleEncode = preprocessing.LabelEncoder()
        data[data.columns[i]] = modleEncode.fit_transform(data[data.columns[i]])



scaler = preprocessing.MinMaxScaler()

scaled_data = scaler.fit_transform(data)

scaled_data = pd.DataFrame(scaled_data ,columns=data.columns)




r = scaled_data.corr()

print(' data correlation : \n ' , r)   


r = scaled_data.corr()

sns.heatmap(r , annot= True)
plt.show()


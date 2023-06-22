
import pandas as pd

data = pd.read_csv('./sample_data/data.csv')

description = data.describe()
print(description)

null_values = data.isnull().sum()
print("Null values:\n", null_values)

data_filled = data.fillna(data.mean())

null_values_filled = data_filled.isnull().sum()
print("Null values after replacing with mean:\n", null_values_filled)

selected_columns = ['Duration', 'Pulse']  

aggregated_data = data[selected_columns].agg(['min', 'max', 'count', 'mean'])
print(aggregated_data)

filtered_data = data[(data['Calories'] >= 500) & (data['Calories'] <= 1000)]


print(filtered_data)

filtered_data = data[(data['Calories'] > 500) & (data['Pulse'] < 100)]


print(filtered_data)

df_modified = data.drop("Maxpulse", axis=1)

print(df_modified)

data.drop("Maxpulse", axis=1, inplace=True)

print(data)

data['Calories'].fillna(0, inplace=True)


data['Calories'] = data['Calories'].astype(int)
print(data.dtypes)

import matplotlib.pyplot as plt

data.plot(kind='scatter', x='Duration', y='Calories')


plt.title('Scatter Plot: Duration vs Calories')
plt.xlabel('Duration')
plt.ylabel('Calories')


plt.show()



import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


glass_data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data', header=None)

glass_data.columns = ['Id', 'RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'Type']

X = glass_data.drop('Type', axis=1)
y = glass_data['Type']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

gnb = GaussianNB()
gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)

from sklearn.metrics import accuracy_score

accuracy = gnb.score(X_test, y_test)

print("Accuracy: {:.2f}".format(accuracy))

from sklearn.metrics import classification_report

report = classification_report(y_test, y_pred)

print(report)
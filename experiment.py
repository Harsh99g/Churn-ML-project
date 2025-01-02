import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, tensorboard
import datetime
import pickle

data = pd.read_csv('Churn_Modelling.csv')
data.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace=True)

leg = LabelEncoder()
data["Gender"] = leg.fit_transform(data['Gender'])
# print(data.columns)
onehot_encoder_geo = OneHotEncoder()
geo_encoder = onehot_encoder_geo.fit_transform(data[['Geography']]).toarray()
print(geo_encoder)
onehot_encoder_geo.get_feature_names_out(['Geography'])
geo_encoded_df = pd.DataFrame(geo_encoder, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))
# print(geo_encoded_df)
data = pd.concat([data.drop('Geography', axis=1), geo_encoded_df], axis=1)
# with open('label_encoder_gender.pkl', 'wb') as file:
#     pickle.dump(leg, file)
#
# with open('onehot_encoder_geo.pkl', 'wb') as file:
#     pickle.dump(onehot_encoder_geo, file)

x = data.drop(['Exited'], axis=1)
y = data['Exited']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
scalar = StandardScaler()
x_train = scalar.fit_transform(x_train)
x_test = scalar.transform(x_test)

# with open('scalar.pkl', 'wb') as file:
#     pickle.dump(scalar, file)

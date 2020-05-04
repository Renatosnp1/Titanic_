# -*- coding: utf-8 -*-

import pandas as pd
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split


df_train = pd.read_csv('train.csv')

df_train.columns = ['PassengerId', 'Survived', 'Classe', 'Nome', 'Sexo', 'Idade',
                    'Num_irmao_esposa','Num_pais_filhos', 'bilhete','tarifa',
                    'Num_cabine', 'Local_embarque']

df_train1 = df_train[['Classe', 'Sexo', 'Idade', 'Local_embarque', 'Survived',]]


# CRIAÇÃO DE DUMMIES e montando DATASET
train = pd.get_dummies(df_train1["Local_embarque"])
sexo_dummies = pd.get_dummies(df_train1["Sexo"])
train["Survived"] = df_train1["Survived"]
train["Idade"] = df_train1["Idade"]
train["Female"] = sexo_dummies["female"]
train["Male"] = sexo_dummies["male"]
train['Idade'].fillna(31, inplace=True)


x_train, x_test, y_train, y_test = train_test_split(train.drop(columns=["Survived"]),
                                            train["Survived"], test_size=0.25,
                                            random_state = 137, stratify=train["Survived"])
model = GaussianNB()
model.fit(x_train, y_train)
model.predict(x_test)
metrics.accuracy_score(y_test, model.predict(x_test))


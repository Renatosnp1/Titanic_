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

def faixaIdade(idd):
    if idd <= 10 :
        return "ate 10"
    elif idd <= 20:
        return "ate 20"
    elif idd <= 30:
        return "ate 30"
    elif idd <= 50:
        return "ate 40"
    elif idd <= 100:
        return "maior de 50"      

train["Idade"] = train['Idade'].apply(faixaIdade)

train["ate 10"] = pd.get_dummies(train['Idade']).iloc[:,0]
train["ate 20"] = pd.get_dummies(train['Idade']).iloc[:,1]
train["ate 30"] = pd.get_dummies(train['Idade']).iloc[:,2]
train["ate 40"] = pd.get_dummies(train['Idade']).iloc[:,3]
train["Maior de 50"] = pd.get_dummies(train['Idade']).iloc[:,4]

train = train.drop(["Idade"], axis = 1)
    
# lista = []
# for i in range(1, 400):        
#     x_train, x_test, y_train, y_test = train_test_split(train.drop(columns=["Survived"]),
#                                                 train["Survived"], test_size=0.20,
#                                                 random_state = i)
#     model = GaussianNB()
#     model.fit(x_train, y_train)
#     model.predict(x_test)
#     lista.append([metrics.accuracy_score(y_test, model.predict(x_test)), i])
# max(lista)

x_train, x_test, y_train, y_test = train_test_split(train.drop(columns=["Survived"]),
                                    train["Survived"], test_size=0.25,
                                    random_state = 137, stratify=train["Survived"])
model = GaussianNB()
model.fit(x_train, y_train)
model.predict(x_test)
metrics.accuracy_score(y_test, model.predict(x_test))
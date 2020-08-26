#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 18:29:17 2020

@author: jackie chan 🙅
"""


# Librerie

import numpy as np # https://it.wikipedia.org/wiki/NumPy
import matplotlib.pyplot as plt # libreria per grafici https://it.wikipedia.org/wiki/Matplotlib
import pandas as pd # https://pandas.pydata.org/


from sklearn.model_selection import train_test_split # utility per dividere il dataset in training set e test set
from sklearn.preprocessing import StandardScaler # utility per portare tutti i valori sulla stessa scala

from tensorflow.keras.layers import Dense # layer di neuroni
from tensorflow import keras # libreria per il deep learning https://keras.io/


# Caricamento del dataset
option_dataset = pd.read_csv('dataset.csv')

# Assegna i nomi delle colonne come indice
option_dataset.head()

# Separazione dataset tra input e output
y = option_dataset[['imp_vol']]
X = option_dataset[['change', 'delta', 'maturity']]

# Divisone train set e test set (80/20)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=100)

# Divisione del test set in test set e validation set (75/25)
X_train,X_val,y_train,y_val=train_test_split(X_train,y_train,test_size=0.25,random_state=100)

# Normalizzazione dei valori
scaler = StandardScaler()
scaler.fit(X_train)

X_scaled_train = scaler.transform(X_train)
X_scaled_vals = scaler.transform(X_val)
X_scaled_test = scaler.transform(X_test)

# Trasformazione dataframe in array
y_train = np.asarray(y_train)
y_val = np.asarray(y_val)
y_test = np.asarray(y_test)

# Crezione modello Deep Learning
# Modello creato con 3 layer da 20 neuroni ciascono con la funzione di attivazione sigmoide
model = keras.models.Sequential([
    Dense(20,activation = "sigmoid",input_shape = (3,)),
    Dense(20,activation = "sigmoid"),
    Dense(20,activation = "sigmoid"),
    Dense(1)
])

# Stampa caratteristiche del modello
model.summary()


# Complie function allows you to choose your measure of loss and optimzer
# For other optimizer, please refer to https://keras.io/optimizers/
# Compilazione con l'ottimizzatore "Adam" https://keras.io/api/optimizers/adam/ che serve per evitare l'overfitting
# Mean absolute error viene utilizzato per calcolare il oss
model.compile(loss = "mae",optimizer = "Adam")


# La funzione ModelCheckpoint viene utilizzata per salvarsi periodicamente su file la versione più "fit" del modello
checkpoint_cb = keras.callbacks.ModelCheckpoint("bs_pricing_model_vFinal.h5",save_best_only = True)

# La funzione EearlyStopping viene utilizzata per fermare il training in caso non ci siano miglioramenti (in questo caso per 2000 epoche)
# Una volta raggiunto lo stop viene ricaricato il modello migliore
early_stopping_cb = keras.callbacks.EarlyStopping(patience = 2000,restore_best_weights = True)

# Training del modello
# La funzione fit server per allenare il modello utilizzando il training dataset, numero di epoche (50000)
# dati di validazione e le funzioni callback per il checkpoint e l'earlystopping

# Anche mettendo 50000 epoche il modello fermerà il training se entro 2000 epoche non vede miglioramenti
history=model.fit(X_scaled_train,y_train[:,0],epochs= 50000,verbose = 1, validation_data=(X_scaled_vals,y_val[:,0]),callbacks=[checkpoint_cb,early_stopping_cb])

# Carica il miglior modello salvato dal ModelCheckpoint
model = keras.models.load_model("bs_pricing_model_vFinal.h5")

# Calcolo del Mean Absolute Error
mae_test = model.evaluate(X_scaled_test,y_test[:,0],verbose=0)
print('Nerual network mean absoluste error on test set:', mae_test)

# Predizione dei dati di test con il modello allenato
model_prediction = model.predict(X_scaled_test)

# Calcolo mean error e standard error della predizione sul dataset di test
mean_error = np.average(model_prediction.T - y_test[:,0])
std_error = np.std(model_prediction.T - y_test[:,0])

# Stampo i risultati
print('Statistics:')
print(" ")
print('Neural Network Statistics:')
print('Mean error on test set vs. option price with noise:',mean_error)
print('Standard deviation of error on test set vs. option price with noise:',std_error)

# Mostro i risultati su un grafico
pd.DataFrame(history.history).plot(figsize=(8,5))
plt.grid(True)
plt.gca().set_ylim(0.1,0.2)
plt.show()

# Salvo l'output del grafico in un csv
output = pd.DataFrame(history.history)
output.to_csv("mae_history.csv")





import tensorflow as tf
from tensorflow import keras
from keras import layers

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

## IMPORT DATASETS
# download dataset
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                'Acceleration', 'Model Year', 'Origin']

raw_dataset = pd.read_csv(url, names=column_names,
                          na_values='?', comment='\t',
                          sep=' ', skipinitialspace=True)

dataset = raw_dataset.copy()

# drop unknown values
dataset.isna().sum()
dataset = dataset.dropna()

# encode values
dataset["Origin"] = dataset["Origin"].map({1: "USA", 2: "Europe", 3:"Japan"})
dataset = pd.get_dummies(dataset, columns=["Origin"], prefix="", prefix_sep="")

# split dataset into testing and training
train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

# label
train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_labels = train_features.pop("MPG")
test_labels = test_features.pop("MPG")


## Normalization
print(train_dataset.describe().transpose()[["mean", "std"]])
print("")

normalizer = keras.layers.Normalization(axis=-1)
normalizer.adapt(np.array(train_features))

first = np.array(train_features[:1])
with np.printoptions(precision=2, suppress=True):
    print("First example:", first)
    print("")
    print("Normalized:", normalizer(first).numpy())
print("")

## LINEAR REGRESSION WITH ONE INPUT
horsepower = np.array(train_features["Horsepower"])
horsepower_normalizer = layers.Normalization(input_shape=[1,], axis=None)
horsepower_normalizer.adapt(horsepower)

# Build model
horsepower_model = keras.Sequential([
    horsepower_normalizer,
    layers.Dense(units=1)
])

horsepower_model.predict([horsepower[:10]])

# Compile
horsepower_model.compile(
    optimizer = keras.optimizers.Adam(learning_rate=0.1),
    loss = "mean_absolute_error"
)

# Train model
history = horsepower_model.fit(
    train_features['Horsepower'],
    train_labels,
    epochs=100,
    verbose=0,
    validation_split = 0.2)

hist = pd.DataFrame(history.history)
hist["epoch"] = history.epoch

def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylim([0, 10])
    plt.xlabel('Epoch')
    plt.ylabel('Error [MPG]')
    plt.legend()
    plt.grid(True) 
    plt.show()

plot_loss(history)

test_results = {}
test_results["horsepower_model"] = horsepower_model.evaluate(
    test_features["Horsepower"],
    test_labels, verbose=0
)

x = tf.linspace(0.0, 250, 251)
y = horsepower_model.predict(x)

def plot_horsepower(x, y):
    plt.scatter(train_features['Horsepower'], train_labels, label='Data')
    plt.plot(x, y, color='k', label='Predictions')
    plt.xlabel('Horsepower')
    plt.ylabel('MPG')
    plt.legend()
    plt.show()

plot_horsepower(x, y)

## LINEAR REGRESSION WITH MULTIPLE INPUTS
linear_model = keras.Sequential([
    normalizer,
    layers.Dense(units=1)
])

linear_model.predict(train_features[:10])
linear_model.layers[1].kernel

linear_model.compile(
    optimizer = keras.optimizers.Adam(learning_rate=0.1),
    loss = "mean_absolute_error"
)

history = linear_model.fit(
    train_features,
    train_labels,
    epochs=100,
    verbose=0,
    validation_split = 0.2)

plot_loss(history)

test_results["linear_model"] = linear_model.evaluate(
    test_features, test_labels, verbose=0)


## REGRESSION USING A DEEP NEURAL NETWORK
def build_and_compile_model(norm):
    model = keras.Sequential([
        norm,
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])

    model.compile(loss='mean_absolute_error',
                    optimizer=tf.keras.optimizers.Adam(0.001))
    return model

# Regression using a DNN and a single input
# compile model
dnn_horsepower_model = build_and_compile_model(horsepower_normalizer)

# train model
history = dnn_horsepower_model.fit(
    train_features["Horsepower"],
    train_labels,
    validation_split = 0.2,
    verbose = 0, epochs = 100)

plot_loss(history)

x = tf.linspace(0.0, 250, 251)
y = dnn_horsepower_model.predict(x)

plot_horsepower(x, y)

test_results["dnn_horsepower"] = dnn_horsepower_model.evaluate(
    test_features["Horsepower"], test_labels,
    verbose = 0)

# Regression using a DNN and a multiple inputs
dnn_model = build_and_compile_model(normalizer)

history = dnn_model.fit(
    train_features,
    train_labels,
    validation_split = 0.2,
    verbose = 0, epochs = 100
)

plot_loss(history)

test_results["dnn_model"] = dnn_model.evaluate(test_features, test_labels, verbose=0)

## PERFORMANCE
pd.DataFrame(test_results, index=["Mean absolute error [MPG]"]).T

# Predictions
test_predictions = dnn_model.predict(test_features).flatten()

a = plt.axes(aspect="equal")
plt.scatter(test_labels, test_predictions)
plt.xlabel("True Values [MPG]")
plt.ylabel("Predictions [MPG]")
lims = [0, 50]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)
plt.show ()

# error distribution
error = test_predictions - test_labels
plt.hist(error, bins=25)
plt.xlabel("Prediction Error [MPG]")
_ = plt.ylabel("Count")
plt.show()
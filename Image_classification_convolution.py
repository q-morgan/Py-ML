import tensorflow as tf
from tensorflow import keras
from keras import datasets, layers, models

import numpy as np
import os
import matplotlib.pyplot as plt

## IMPORT DATASETS
data = datasets.cifar10
(train_images, train_labels), (test_images, test_labels) = data.load_data()

class_names = ["Airplane", "Automobile", "Bird", "Cat", "Deer", 
                "Dog", "Frog", "Horse", "Ship", "Truck"]

# Conv2D = combined with the layer input to produce a tensor of outputs
# MaxPooling2D = ?
model = models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation="relu", input_shape = (32,32,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation="relu"))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation="relu"))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dense(10))

model.summary()

## COMPILE
model.compile(optimizer="adam",
            loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics = ["accuracy"])


## TRAIN
history = model.fit(train_images, train_labels, epochs=10,
                    validation_data=(test_images, test_labels))


## EVALUATE
def evalute():
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')

    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
    print(test_acc)
    plt.show

evalute()


import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

data = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = data.load_data()

class_names = ["T-Shirts", "Trousers", "Pullover", "Dress", "Coat", 
"Sandal", "Sandals", "Sneaker", "Bag", "Ankle Boot"]

# find the colour range
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

# the colour range was 0-255 so it is divided by 255 to make the range 0-1
train_images = train_images / 255.0
test_images = test_images / 255.0

# check format of the first 25 images 
plt.figure(figsize = (10, 10))
for i in range(25):
    plt.subplot(5, 5 , i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

# Flatten = transform image from matrix to single array
#
# [1,2], 
# [5,6],   =   [1,2,5,6,3,4]
# [3,4]
#
# Dense = neural network layer
# Dense(128) = 128 nodes
model = keras.Sequential([
    keras.layers.Flatten(input_shape = (28,28)),
    keras.layers.Dense(128, activation = "relu"),
    keras.layers.Dense(10)
])

# losses = measures how accurate the model is during training
# Optimizer = how the model is updated based on the data and losses
# Metrics = used to monitor the training and testing stages
model.compile(optimizer = "adam", 
            loss = keras.losses.SparseCategoricalCrossenttrophy(from_logists = True),
            metrics = ["accuracy"])

model.fit(train_images, train_labels, epochs = 10)

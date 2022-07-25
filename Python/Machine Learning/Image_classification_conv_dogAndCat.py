import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow import keras
from keras import layers
from keras import Model
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

# load data
base_dir = "/Users/qmorgan/Documents/GitHub/Tutorials/Python/Machine Learning/tmp/cats_and_dogs_filtered"
train_dir = os.path.join(base_dir, "train")
validation_dir = os.path.join(base_dir, "validation")

train_cats_dir = os.path.join(train_dir, "cats")
train_dogs_dir = os.path.join(train_dir, "dogs")
validation_cats_dir = os.path.join(train_dir, "cats")
validation_dogs_dir = os.path.join(train_dir, "dogs")

train_cat_fnames = os.listdir(train_cats_dir)
train_dog_fnames = os.listdir(train_dogs_dir)
train_dog_fnames.sort()

# test data
print('total training cat images:', len(os.listdir(train_cats_dir)))
print('total training dog images:', len(os.listdir(train_dogs_dir)))
print('total validation cat images:', len(os.listdir(validation_cats_dir)))
print('total validation dog images:', len(os.listdir(validation_dogs_dir)))

def explore_and_test_images():
    nrows = 4
    ncols = 4
    pic_index = 0

    fig = plt.gcf()
    fig.set_size_inches(ncols*4, nrows*4)

    pic_index += 8
    next_cat_pix = [os.path.join(train_cats_dir, fname) for fname in train_cat_fnames[pic_index-8:pic_index]]
    next_dog_pix = [os.path.join(train_dogs_dir, fname) for fname in train_dog_fnames[pic_index-8:pic_index]]

    for i, img_path in enumerate(next_cat_pix+next_dog_pix):
        sp = plt.subplot(nrows, ncols, i+1)
        sp.axis("Off")
        img = mpimg.imread(img_path)
        plt.imshow(img)

    plt.show()

# explore_and_test_images()

# build
# input feature map is 150x150x3:
# 150x150 for the image pixels, and 3 for the three color channels: R, G, and B
img_input = keras.layers.Input(shape=(150,150,3))

# This convolution extracts 16 filters that are 3x3
x = layers.Conv2D(16, 3, activation="relu")(img_input)
x = layers.MaxPooling2D(2)(x)

# this convolution extracts 32 filters that are 3x3
x = layers.Conv2D(32, 3, activation="relu")(x)
x = layers.MaxPooling2D(2)(x)

# this convolution extracts 64 filters that are 3x3
x = layers.Conv2D(64, 3, activation="relu")(x)
x = layers.MaxPooling2D(2)(x)
# Conv2D = combined with the layer input to produce a filtered output
#
# [3, 5, 2, 8, 1]         Filter
# [9, 7, 5, 4, 3]       [1, 0, 0]       [3x1, 5x0, 2x0]                     [25, 13, 17]
# [2, 0, 6, 1, 6]   +   [1, 1, 0]   =   [9x1, 7x1, 2x0] for each 3x3    =   [18, 22, 14]
# [6, 3, 7, 9, 2]       [0, 0, 1]       [2x0, 0x0, 6x1]                     [20, 15, 23]
# [1, 4, 9, 5, 1]                       3+0+0+9+7+0+0+0+6 = 25
#
# MaxPooling2D = extract maximum value from input
#
# [7, 3, 5, 2]      [7, 3][5, 2]
# [8, 7, 1, 6]  =   [8, 7][1, 6]    =   [8, 6]
# [4, 9, 3, 9]      [4, 9][3, 9]        [9, 9]
# [0, 8, 4, 5]      [0, 8][4, 5]
# 

x = layers.Flatten()(x)
x = keras.layers.Dense(512, activation="relu")(x)
output = keras.layers.Dense(1, activation="sigmoid")(x)

# create model
model = Model(img_input, output)

model.summary()

model.compile(loss="binary_crossentropy",
            optimizer = RMSprop(learning_rate = 0.001),
            metrics = ["acc"])

train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size = (150, 150),
    batch_size = 20,
    class_mode = "binary"
)

validation_generator = val_datagen.flow_from_directory(
    validation_dir,
    target_size = (150, 150),
    batch_size = 20,
    class_mode = "binary"
)

# training
history = model.fit(
    train_generator,
    steps_per_epoch = 100, # 2000 images = batch_size * steps_per_epochs
    epochs = 15,
    validation_data = validation_generator,
    validation_steps = 50, # 1000 images = batch_size * validation_steps
    verbose = 2
)

# visualise
successive_outputs = [layer.output for layer in model.layers[1:]]
visualization_model = Model(img_input, successive_outputs)

cat_img_files = [os.path.join(train_cats_dir, f) for f in train_cat_fnames]
dog_img_files = [os.path.join(train_dogs_dir, f) for f in train_dog_fnames]
img_path = random.choice(cat_img_files + dog_img_files)

img = load_img(img_path, target_size=(150, 150))
x = img_to_array(img)
x = x.reshape((1,) + x.shape)

x /= 255

successive_feature_maps = visualization_model.predict(x)
layer_names = [layer.name for layer in model.layers[1:]]

for layer_name, feature_map in zip(layer_names, successive_feature_maps):
    if len(feature_map.shape) == 4:
        n_features = feature_map.shape[-1]
        size = feature_map.shape[1]
        display_grid = np.zeros((size, size * n_features))
        for i in range(n_features):
            x = feature_map[0, :, :, i]
            x -= x.mean()
            x /= x.std()
            x *= 64
            x += 128
            x = np.clip(x, 0, 255).astype("uint8")
            display_grid[:, i * size : (i+1) * size] = x
        scale = 20./n_features
        plt.figure(figsize=(scale*n_features,scale))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect="auto", cmap = "viridis")

acc = history.history['acc']
val_acc = history.history['val_acc']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc)
plt.plot(epochs, val_acc)
plt.title('Training and validation accuracy')
plt.show()

plt.figure()
plt.plot(epochs, loss)
plt.plot(epochs, val_loss)
plt.title('Training and validation loss')
plt.show()
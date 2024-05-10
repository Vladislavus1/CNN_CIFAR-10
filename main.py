import tensorflow as tf
import PIL.Image as Image
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

os.environ["TF_ENABLE_ONEDNN_OPTS"] = '0'

keras = tf.keras
cifar10 = keras.datasets.cifar10
Input = keras.layers.Input
Dense = keras.layers.Dense
Flatten = keras.layers.Flatten
Model = keras.models.Model
Adam = keras.optimizers.Adam
VGG19 = keras.applications.vgg19.VGG19
preprocess_input = keras.applications.vgg19.preprocess_input
image = keras.preprocessing.image
ImageDataGenerator = keras.preprocessing.image.ImageDataGenerator
ModelCheckpoint = keras.callbacks.ModelCheckpoint

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

num_samples = 5000
x_train_subset, _, y_train_subset, _ = train_test_split(x_train, y_train, train_size=num_samples, random_state=42)

x_train_subset = x_train_subset / 255.0

IMAGE_SIZE = [32, 32, 3]

vgg = VGG19(include_top=False, input_shape=IMAGE_SIZE, weights='imagenet')

print(vgg.summary())

for layer in vgg.layers:
    layer.trainable = False

x = Flatten()(vgg.output)

prediction = Dense(10, activation='softmax')(x)


model = Model(inputs=vgg.input, outputs=prediction)

print(model.summary())

adam = Adam()
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])

his = model.fit(x_train_subset, y_train_subset,
                validation_split=0.2,
                epochs=20,
                batch_size=32,
                verbose=2)

plt.plot(his.history['loss'])
plt.plot(his.history['val_loss'])
plt.savefig('graphic.png')

img = Image.open('car.jpg')
img = np.array(img)
x = keras.applications.vgg16.preprocess_input(img)
x = np.expand_dims(x, axis=0)
res = model.predict(x)
res = int(np.argmax(res))
print(res)




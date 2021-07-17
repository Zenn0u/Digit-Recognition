import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.keras.engine import input_spec

# Data

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# Model

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(units=256, activation='relu'))
model.add(tf.keras.layers.Dense(units=128, activation='relu'))
model.add(tf.keras.layers.Dense(units=10, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

epoch = 100
batch_size = 128

model.fit(x_train, y_train, epochs=epoch, validation_data=(x_test, y_test), batch_size=batch_size)

loss, accuracy = model.evaluate(x_test, y_test)
print(accuracy)
print(loss)

model.save("mnist.model")

# Testing

for i in range(0,10):
    img = cv2.imread(f"{i}.png")[:,:,0]
    img = np.invert(np.array([img]))
    prediction = model.predict(img)
    print(f"Number is {i} and the prediction is {np.argmax(prediction)}")
    # plt.imshow(img[0], cmap=plt.cm.binary)
    # plt.show()

x_pred = model.predict([x_test])

for i in range(0,100):
    print(np.argmax[i])
    plt.imshow(x_test[i])
    plt.show()
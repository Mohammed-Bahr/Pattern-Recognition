import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2

# mnist = tf.keras.datasets.mnist
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
"""
x_train: 60000 images, each 28x28 pixels -> (60000, 28, 28)
y_train: 60000 labels (0-9) -> (60000,)
x_test: 10000 images, each 28x28 pixels -> (10000, 28, 28)
y_test: 10000 labels (0-9) -> (10000,)
"""

# x_train = tf.keras.utils.normalize(x_train, axis=1)
# x_test = tf.keras.utils.normalize(x_test, axis=1)

# model = tf.keras.models.Sequential()
# model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
# model.add(tf.keras.layers.Dense(128, activation='relu'))
# model.add(tf.keras.layers.Dropout(0.3))
# model.add(tf.keras.layers.Dense(128, activation='relu'))
# model.add(tf.keras.layers.Dropout(0.3))
# model.add(tf.keras.layers.Dense(10, activation='softmax'))

# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# model.fit(x_train, y_train, epochs=3)

# model.save('handwritten.keras')

model = tf.keras.models.load_model('handwritten.keras')
# loss , accuracy = model.evaluate(x_test, y_test)
# print("loss is : " + str(loss))
# print("accuracy is : " + str(accuracy))
  


imageNumber = 1
while os.path.isfile(f"samples/digit{imageNumber}.png"):
    try:
        img = cv2.imread(f"samples/digit{imageNumber}.png")[:,:,0]
        img = np.invert(np.array([img]))
        img = img / 255.0
        prediction = model.predict(img)
        print(f"This digit is probably a {np.argmax(prediction)}")
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
    except Exception as e:
        print("Error! -> " + str(e))  

    finally:
        imageNumber += 1
# import os
# import numpy as np
# import matplotlib.pyplot as plt
# import tensorflow as tf
# import cv2

# # mnist = tf.keras.datasets.mnist
# # (x_train, y_train), (x_test, y_test) = mnist.load_data()
# """
# x_train: 60000 images, each 28x28 pixels -> (60000, 28, 28)
# y_train: 60000 labels (0-9) -> (60000,)
# x_test: 10000 images, each 28x28 pixels -> (10000, 28, 28)
# y_test: 10000 labels (0-9) -> (10000,)
# """

# # x_train = tf.keras.utils.normalize(x_train, axis=1)
# # x_test = tf.keras.utils.normalize(x_test, axis=1)

# # model = tf.keras.models.Sequential()
# # model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
# # model.add(tf.keras.layers.Dense(128, activation='relu'))
# # model.add(tf.keras.layers.Dropout(0.3))
# # model.add(tf.keras.layers.Dense(128, activation='relu'))
# # model.add(tf.keras.layers.Dropout(0.3))
# # model.add(tf.keras.layers.Dense(10, activation='softmax'))

# # model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# # model.fit(x_train, y_train, epochs=3)

# # model.save('handwritten.keras')

# model = tf.keras.models.load_model('handwritten.keras')
# # loss , accuracy = model.evaluate(x_test, y_test)
# # print("loss is : " + str(loss))
# # print("accuracy is : " + str(accuracy))
  


# imageNumber = 1
# while os.path.isfile(f"samples/digit{imageNumber}.png"):
#     try:
#         img = cv2.imread(f"samples/digit{imageNumber}.png")[:,:,0]
#         img = np.invert(np.array([img]))
#         img = img / 255.0
#         prediction = model.predict(img)
#         print(f"This digit is probably a {np.argmax(prediction)}")
#         plt.imshow(img[0], cmap=plt.cm.binary)
#         plt.show()
#     except Exception as e:
#         print("Error! -> " + str(e))  

#     finally:
#         imageNumber += 1







import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2

# ==========================================
# 1. TRAINING SECTION (Run this once)
# ==========================================
"""
Uncomment this block to train the model properly.
You MUST train it using simple / 255.0 normalization 
so it matches your custom image processing later.
"""
# mnist = tf.keras.datasets.mnist
# (x_train, y_train), (x_test, y_test) = mnist.load_data()

# # FIX: Use simple division for normalization instead of tf.keras.utils.normalize
# x_train = x_train / 255.0
# x_test = x_test / 255.0

# model = tf.keras.models.Sequential()
# model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
# model.add(tf.keras.layers.Dense(128, activation='relu'))
# model.add(tf.keras.layers.Dropout(0.3))
# model.add(tf.keras.layers.Dense(128, activation='relu'))
# model.add(tf.keras.layers.Dropout(0.3))
# model.add(tf.keras.layers.Dense(10, activation='softmax'))

# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# model.fit(x_train, y_train, epochs=5) # Bumped to 5 epochs for a bit more accuracy

# model.save('handwritten.keras')


# ==========================================
# 2. PREDICTION SECTION
# ==========================================

# Load the trained model
model = tf.keras.models.load_model('handwritten.keras')

imageNumber = 1

# Loop through samples/digit1.png, samples/digit2.png, etc.
while os.path.isfile(f"samples/digit{imageNumber}.png"):
    try:
        # 1. Load the image strictly in grayscale mode
        img = cv2.imread(f"samples/digit{imageNumber}.png", cv2.IMREAD_GRAYSCALE)
        
        # 2. Safety resize: ensure it is exactly 28x28 pixels
        img = cv2.resize(img, (28, 28))
        
        # 3. Invert colors (Assumes you drew black text on a white background)
        # NOTE: If you drew white text on a black background, delete the line below!
        img = np.invert(img)
        
        # 4. Normalize the exact same way the model was trained
        img = img / 255.0
        
        # 5. The model expects a batch (a list of images), so we wrap it in an array
        img_batch = np.array([img])
        
        # Predict
        prediction = model.predict(img_batch, verbose=0) # verbose=0 hides the progress bar
        predicted_digit = np.argmax(prediction)
        
        print(f"Image {imageNumber}: This digit is probably a {predicted_digit}")
        
        # Show the image visually
        plt.imshow(img_batch[0], cmap=plt.cm.binary)
        plt.title(f"Predicted: {predicted_digit}")
        plt.show()
        
    except Exception as e:
        print(f"Error on image {imageNumber}! -> {str(e)}")  
        
    finally:
        imageNumber += 1
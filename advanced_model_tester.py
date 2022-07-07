import os
import sys
import cv2

import numpy as np

from tensorflow import keras
from matplotlib import pyplot as plt

def makePrediction(ml, img):
    predicted = ml.predict(np.array([img]), verbose = 0)
    plotImage(img, np.argmax(predicted))

def plotImage(img, pred):
    plt.figure()
    plt.imshow(img)
    plt.title("Predicted: " + str(pred))
    plt.grid(False)
    plt.show()

def convertImage(img):
    raw_img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)

    new_img = cv2.resize(raw_img, (28, 28), interpolation = cv2.INTER_LINEAR)
    new_img = cv2.bitwise_not(new_img)

    return new_img

PATH = input("Enter where to import model from: ")

if os.path.isdir(PATH) == False:
    print("Invalid path")
    sys.exit(0)

name = "Model"
new = os.path.join(PATH, name)

if os.path.isdir(new) == False:
    print("No \"Model\" folder in directory")
    sys.exit(0)

model = keras.models.load_model(new)

model.summary()

img = input("Choose which image to to predict: ")

if os.path.isfile(img) == False:
    print("File does not exist")
    sys.exit(0)

x = convertImage(img)
makePrediction(model, x)



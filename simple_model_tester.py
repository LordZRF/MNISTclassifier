import os
import sys

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

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

n = int(input("Pick a number between 0 and {}: ".format(x_test.shape[0] - 1)))
if n >= x_test.shape[0] or n < 0:
    print("Invalid number")
    sys.exit(0)
makePrediction(model, x_test[n])

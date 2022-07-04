import sys
import os

from tensorflow import keras

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation = 'relu', input_shape = x_train.shape[1:]),
    keras.layers.MaxPool2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation = 'relu'),
    keras.layers.MaxPool2D((2, 2)),
    keras.layers.Dropout(0.3),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation = 'relu'),
    keras.layers.Dense(10, activation = 'softmax')
])

model.summary()

model.compile(optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])

model.fit(x_train, y_train, epochs = 10)

print("\nTraining complete\n")

_, test_acc = model.evaluate(x_test,  y_test, verbose = 1)

print("\nTest accuracy: ", test_acc * 100, "% !\n", sep = '')

ans = input("Do you want to save the model?(Y/n)")

if ans != "Y":
    sys.exit(0)

PATH = input("Enter where to save the model: ")

if os.path.isdir(PATH) == False:
    print("Invalid path")
    sys.exit(0)

name = "Model"
new = os.path.join(PATH, name)
os.mkdir(new)

model.save(new)

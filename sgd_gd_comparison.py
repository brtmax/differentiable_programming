import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import SGD, Adam
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

# Preprocess data
x_train = x_train / 255.0
x_val = x_val / 255.0
x_test = x_test / 255.0

# Define the model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Compile the model with SGD optimizer
sgd = SGD(lr=0.01)
model.compile(loss='sparse_categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Train the model with SGD
print("Training with SGD...")
start_time = time.time()
sgd_history = model.fit(x_train, y_train, batch_size=32, epochs=50, validation_data=(x_val, y_val))
sgd_time = time.time() - start_time

# Compile the model with GD optimizer
gd = Adam(lr=0.01)
model.compile(loss='sparse_categorical_crossentropy', optimizer=gd, metrics=['accuracy'])

# Train the model with GD
print("Training with GD...")
start_time = time.time()
gd_history = model.fit(x_train, y_train, batch_size=len(x_train), epochs=50, validation_data=(x_val, y_val))
gd_time = time.time() - start_time

# Compare training histories
sgd_acc = sgd_history.history['accuracy']
gd_acc = gd_history.history['accuracy']

# Plot the training accuracy
epochs = range(1, len(sgd_acc) + 1)
plt.plot(epochs, sgd_acc, 'r', label=f'SGD (Time: {sgd_time:.2f}s)')
plt.plot(epochs, gd_acc, 'b', label=f'GD (Time: {gd_time:.2f}s)')
plt.title('Training Accuracy Comparison')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


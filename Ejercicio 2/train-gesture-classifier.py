import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# Load dataset
X = np.load('Ejercicio 2\\rps_dataset.npy')
y = np.load('Ejercicio 2\\rps_labels.npy')

# Normalize data
X = X / np.max(X)

# One-hot encode the labels
y = to_categorical(y, num_classes=3)

# Define the model
model = Sequential([
    Dense(64, input_shape=(42,), activation='relu'),
    Dense(64, activation='relu'),
    Dense(3, activation='softmax')  # Output layer with 3 classes
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X, y, epochs=50, batch_size=8)

# Save the trained model
model.save('Ejercicio 2\\rps_model.h5')

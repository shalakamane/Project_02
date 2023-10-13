import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, InputLayer, BatchNormalization, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import time

# Create data generators
imagegen = ImageDataGenerator()

train = imagegen.flow_from_directory(
    "/home/user2/cmake-3.21.3/project/ImageProcessing/Imagenet/imagenette2-320/train",
    class_mode="categorical", shuffle=False, batch_size=32, target_size=(224, 224))

val = imagegen.flow_from_directory(
    "/home/user2/cmake-3.21.3/project/ImageProcessing/Imagenet/imagenette2-320/val",
    class_mode="categorical", shuffle=False, batch_size=32, target_size=(224, 224))

# Build the model
model = Sequential()
model.add(InputLayer(input_shape=(224, 224, 3)))

model.add(Conv2D(25, (5, 5), activation='relu', strides=(1, 1), padding='same'))
model.add(MaxPool2D(pool_size=(2, 2), padding='same'))

model.add(Conv2D(50, (5, 5), activation='relu', strides=(2, 2), padding='same'))
model.add(MaxPool2D(pool_size=(2, 2), padding='same'))
model.add(BatchNormalization())

model.add(Conv2D(70, (3, 3), activation='relu', strides=(2, 2), padding='same'))
model.add(MaxPool2D(pool_size=(2, 2), padding='valid'))
model.add(BatchNormalization())

model.add(Flatten())
model.add(Dense(units=100, activation='relu'))
model.add(Dense(units=100, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(units=10, activation='softmax'))

# Compile model
model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])

# Record start time
start_time = time.time()

# Fit the model
history = model.fit(train, epochs=30, validation_data=val)

# Calculate training time
end_time = time.time()
training_time = end_time - start_time

# Print training time and accuracy
print(f"Training Time: {training_time:.2f} seconds")
print(f"Final Training Accuracy: {history.history['accuracy'][-1]:.4f}")
print(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")

# Save the trained model
model.save("trained_model.h5")
print("Model saved as trained_model.h5")


import sys
sys.stdout.reconfigure(encoding='utf-8')

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os

# ✅ Step 1: Define Image Parameters
IMG_SIZE = 150  # Reducing image size to avoid memory issues
BATCH_SIZE = 32  # Load images in batches
DATASET_PATH = "D:/Github/Projects/ChetanSir/AIML/Training/"  # Update with your dataset directory

# ✅ Step 2: Use ImageDataGenerator for Efficient Data Loading & Augmentation
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,  # Normalize images
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # 80% training, 20% validation
)

train_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

val_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)

# ✅ Step 3: Build the CNN Model
model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D(2, 2),
    
    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),  # Helps reduce overfitting
    Dense(train_generator.num_classes, activation="softmax")  # Output layer
])

# ✅ Step 4: Compile the Model
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ✅ Step 5: Train the Model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10  # Adjust based on your dataset size
)

# ✅ Step 6: Evaluate the Model
loss, accuracy = model.evaluate(val_generator)
print(f"\n🔥 Final Model Accuracy: {accuracy * 100:.2f}%\n")

# ✅ Step 7: Plot Accuracy & Loss Graphs
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Model Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Model Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.show()

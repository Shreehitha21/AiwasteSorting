import tensorflow as tf # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.applications import MobileNetV2 # pyright: ignore[reportMissingImports]
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input # type: ignore
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
import matplotlib.pyplot as plt
import os

# --- Configuration ---
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10

# [FIX 1] This path now matches your folder structure exactly
DATA_DIR = 'dataset/Garbage classification/Garbage classification' 

MODEL_SAVE_PATH = 'model/waste_sorter.h5'

# Define the classes
CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# --- 1. Data Preprocessing and Augmentation ---
print("Setting up data generators...")

# Augmentation for the training set
# We will use 20% of all data for validation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2 # Use 20% of data for validation
)

# [FIX 2] Point the training generator to the main data folder
train_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=CLASSES,
    subset='training' # Set as training data
)

# [FIX 3] Point the validation generator to the main data folder
validation_generator = train_datagen.flow_from_directory(
    DATA_DIR, 
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=CLASSES,
    subset='validation' # Set as validation data
)

print(f"Found {train_generator.samples} training images.")
print(f"Found {validation_generator.samples} validation images.")

# [FIX 4] Removed the separate "Test" generator, as this dataset doesn't have it.
# We will rely on the validation accuracy.

# --- 2. Build the Model (Transfer Learning) ---
print("Building model...")

# Load pre-trained MobileNetV2
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False, # Exclude the final classification layer
    input_tensor=Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
)

# Freeze the base model layers
base_model.trainable = False

# Add our custom classification head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(len(CLASSES), activation='softmax')(x)

# Combine base model and new head
model = Model(inputs=base_model.input, outputs=predictions)

# --- 3. Compile the Model ---
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# --- 4. Train the Model ---
print("Starting model training...")

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE,
    epochs=EPOCHS
)

print("Training finished.")

# --- 5. Save Model and Plot History ---

# Save the final model
print(f"Saving model to {MODEL_SAVE_PATH}...")
model.save(MODEL_SAVE_PATH)
print("Model saved successfully.")

# Plot training & validation accuracy/loss
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(EPOCHS)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.savefig('training_history.png')
print("Training history plot saved as 'training_history.png'")

# [FIX 5] Removed evaluation on the non-existent test set.
print("--- Training Complete ---")
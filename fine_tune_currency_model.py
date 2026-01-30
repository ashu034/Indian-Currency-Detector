import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

# --------------------------
# Step 1: Paths
# --------------------------
train_dir = "dataset_split/train"   # Folder with subfolders for each note class
val_dir = "dataset_split/val"       # Validation folder
model_path = "models/currency_detector_mobilenetv2.h5"  # Path to your pre-trained model
finetuned_model_path = "currency_model_finetuned.h5"

# --------------------------
# Step 2: Data Augmentation
# --------------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.05,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

class_names = list(train_generator.class_indices.keys())
print("Detected classes:", class_names)

# --------------------------
# Step 3: Load and Modify Model
# --------------------------
base_model = load_model(model_path)

# Freeze all layers except last 5
for layer in base_model.layers[:-5]:
    layer.trainable = False

# Add new layers with unique names
x = base_model.layers[-1].output
x = Dense(128, activation='relu', name='dense_finetune')(x)
x = Dropout(0.3, name='dropout_finetune')(x)
output = Dense(len(class_names), activation='softmax', name='output_finetune')(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(optimizer=Adam(learning_rate=1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# --------------------------
# Step 4: Fine-tune Model
# --------------------------
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=15,  # increase if needed
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_steps=val_generator.samples // val_generator.batch_size
)

# --------------------------
# Step 5: Save Fine-tuned Model
# --------------------------
model.save(finetuned_model_path)
print(f"Fine-tuned model saved at: {finetuned_model_path}")

# --------------------------
# Step 6: Prediction Function
# --------------------------
def predict_currency(img_path):
    from tensorflow.keras.preprocessing import image
    import numpy as np

    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = tf.expand_dims(img_array, 0)  # Add batch dimension

    pred = model.predict(img_array)
    confidence = pred.max()
    label = class_names[pred.argmax()]

    if confidence < 0.6:
        return "Prediction uncertain"
    return f"Prediction: {label} ({confidence*100:.2f}%)"

# Example usage:
# print(predict_currency("test_images/500.jpg"))

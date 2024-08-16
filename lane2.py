import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Paths to your dataset folders
xtrain_path = '/content/drive/MyDrive/Lightray/ImgeSegmentation/train'
ytrain_path = '/content/drive/MyDrive/Lightray/ImgeSegmentation/train_y'
xval_path = '/content/drive/MyDrive/Lightray/ImgeSegmentation/val_x'
yval_path = '/content/drive/MyDrive/Lightray/ImgeSegmentation/val_y'

# Load images
def load_images_from_folder(folder, target_size=(256, 256)):
    images = []
    filenames = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        if os.path.isfile(img_path):
            img = load_img(img_path, target_size=target_size)
            img_array = img_to_array(img) / 255.0
            images.append(img_array)
            filenames.append(filename)
    print(f"Loaded {len(images)} images from {folder}")
    return np.array(images), filenames

print("Loading training images...")
xtrain_images, xtrain_filenames = load_images_from_folder(xtrain_path)
print("Loading training labels...")
ytrain_images, _ = load_images_from_folder(ytrain_path)
print("Loading validation images...")
xval_images, xval_filenames = load_images_from_folder(xval_path)
print("Loading validation labels...")
yval_images, _ = load_images_from_folder(yval_path)

# Define U-Net model
def unet_model(input_shape):
    inputs = Input(input_shape)

    # Encoding path
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    c4 = Conv2D(512, (3, 3), activation='relu', padding='same')(c4)
    p4 = MaxPooling2D((2, 2))(c4)

    # Bottleneck
    c5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(p4)
    c5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(c5)

    # Decoding path
    u6 = UpSampling2D((2, 2))(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(512, (3, 3), activation='relu', padding='same')(u6)
    c6 = Conv2D(512, (3, 3), activation='relu', padding='same')(c6)

    u7 = UpSampling2D((2, 2))(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(256, (3, 3), activation='relu', padding='same')(u7)
    c7 = Conv2D(256, (3, 3), activation='relu', padding='same')(c7)

    u8 = UpSampling2D((2, 2))(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(128, (3, 3), activation='relu', padding='same')(u8)
    c8 = Conv2D(128, (3, 3), activation='relu', padding='same')(c8)

    u9 = UpSampling2D((2, 2))(c8)
    u9 = concatenate([u9, c1])
    c9 = Conv2D(64, (3, 3), activation='relu', padding='same')(u9)
    c9 = Conv2D(64, (3, 3), activation='relu', padding='same')(c9)

    outputs = Conv2D(3, (1, 1), activation='softmax')(c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model

input_shape = (256, 256, 3)  # Adjust based on your image size
model = unet_model(input_shape)

# Compile the model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
print("Starting training...")
history = model.fit(xtrain_images, ytrain_images, validation_data=(xval_images, yval_images), epochs=10, batch_size=8)

# Print accuracy
train_accuracy = history.history['accuracy'][-1]
val_accuracy = history.history['val_accuracy'][-1]
print(f"Final Training Accuracy: {train_accuracy:.4f}")
print(f"Final Validation Accuracy: {val_accuracy:.4f}")

# Save the model
model_dir = '/content/drive/MyDrive/Lightray/ImgeSegmentation'
model_save_path = os.path.join(model_dir, 'unet_model.h5')
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
model.save(model_save_path)
print(f"Model saved to {model_save_path}")

# Make predictions
def predict_and_save(model, images, filenames, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    print("Making predictions...")
    predictions = model.predict(images)
    for i, (pred, filename) in enumerate(zip(predictions, filenames)):
        pred_image = np.argmax(pred, axis=-1).astype(np.uint8)
        output_path = os.path.join(output_folder, f'pred_{i}.png')
        plt.imsave(output_path, pred_image)
        print(f"Saved prediction {i} to {output_path}")

output_folder = '/content/drive/MyDrive/Lightray/ImgeSegmentation/output'
predict_and_save(model, xval_images, xval_filenames, output_folder)
print("Predictions complete and saved.")

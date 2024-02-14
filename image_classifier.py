import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


def train_model(real_folder, fake_folder, img_width=150, img_height=150, batch_size=32, epochs=10):
    # Create DataFrame for real images
    real_files = [os.path.join(real_folder, file) for file in os.listdir(real_folder)]
    real_labels = ['1'] * len(real_files)  # Label '1' for real images
    real_df = pd.DataFrame({'file': real_files, 'label': real_labels})

    # Create DataFrame for fake images
    fake_files = [os.path.join(fake_folder, file) for file in os.listdir(fake_folder)]
    fake_labels = ['0'] * len(fake_files)  # Label '0' for fake images
    fake_df = pd.DataFrame({'file': fake_files, 'label': fake_labels})

    # Combine the DataFrames
    full_df = pd.concat([real_df, fake_df], ignore_index=True)

    # Shuffle the DataFrame
    full_df = full_df.sample(frac=1).reset_index(drop=True)

    # Create ImageDataGenerator instance for training data
    train_datagen = ImageDataGenerator(rescale=1./255)

    # Generate batches of augmented data from the DataFrame
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=full_df,
        x_col='file',
        y_col='label',
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary'
    )

    # Calculate steps_per_epoch
    steps_per_epoch = train_generator.samples // batch_size
    if train_generator.samples % batch_size != 0:
        steps_per_epoch += 1

    # Build the CNN model
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs
    )

    # Save the trained model
    model.save("image_classifier_model.h5")


def predict_image(image_path):
    # Load the trained model
    model = load_model("image_classifier_model.h5")

    img = image.load_img(image_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    prediction = model.predict(img_array)
    if prediction[0] < 0.5:
        return "Real"
    else:
        return "Fake"


if __name__ == "__main__":
    # Paths to the real and fake image folders
    real_folder = "static\images\Real"
    fake_folder = "static\images\Fake"

    # Train the model using the specified paths
    train_model(real_folder, fake_folder)

    # Path to the image you want to predict
    image_path = "static/images/test_image.jpg"


    # Predict the image using the trained model
    prediction = predict_image(image_path)
    print(f"Prediction for {image_path}: {prediction}")

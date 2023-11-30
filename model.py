import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
import matplotlib.pyplot as plt

data_dir = "/home/chirag/sketchsync/dataset"

# Validate data directory
assert os.path.isdir(data_dir)
for path in os.listdir(data_dir):
    full_path = os.path.join(data_dir, path)
    if not os.path.isdir(full_path):
        print(f"Removing invalid file {full_path}")
        os.remove(full_path)

print("Dataset directory validated")

def load_images(directory):
    images = []
    labels = []

    for label_folder in os.listdir(directory):
        label_folder_path = os.path.join(directory, label_folder)

        for image_file in os.listdir(label_folder_path):
            if not image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            image_path = os.path.join(label_folder_path, image_file)

            try:
                image = Image.open(image_path).convert('L').resize((28, 28))
                images.append(np.array(image))
                labels.append(label_folder)
            except:
                print(f"Error reading {image_path}, skipping file...")

    return np.array(images), np.array(labels)

images, labels = load_images(data_dir)
images = images / 255.0

train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)

label_encoder = LabelEncoder()
numerical_labels = label_encoder.fit_transform(train_labels)
one_hot_labels = to_categorical(numerical_labels, num_classes=19)  # Assuming 19 classes

unique_test_labels = set(test_labels)
print("Unique Test Labels:", unique_test_labels)

test_numerical_labels = label_encoder.transform(test_labels)
test_one_hot_labels = to_categorical(test_numerical_labels, num_classes=19)

train_labels = one_hot_labels
test_labels = test_one_hot_labels
train_images = train_images.reshape(-1, 28, 28, 1)  # Reshape data to fit the model
test_images = test_images.reshape(-1, 28, 28, 1)

model = Sequential()

# Model layers
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(19, activation='softmax'))  # 19 classes

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train
history = model.fit(
    train_images, train_labels,
    epochs=60,
    batch_size=32,
    validation_data=(test_images, test_labels)
)


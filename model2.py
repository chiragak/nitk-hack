# Import necessary libraries
import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
data_dir = r"C:\Users\SHAMANTH\Desktop\Magic-Chalk-main\dataset1"

def load_images(directory):
    images = []
    labels = []
    for label_folder in os.listdir(directory):
        label_folder_path = os.path.join(directory, label_folder)

        # Check if it's a directory
        if os.path.isdir(label_folder_path):
            for image_file in os.listdir(label_folder_path):
                image_path = os.path.join(label_folder_path, image_file)

                # Check if it's a file and an image (assuming image files have extensions like .jpg, .png, etc.)
                if os.path.isfile(image_path) and image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    #pre-process images
                    image = Image.open(image_path).convert('L')  # grayscale
                    image = image.resize((28, 28))
                    images.append(np.array(image))
                    labels.append(label_folder)

    return np.array(images), np.array(labels)
print("Started loading")
images, labels = load_images(data_dir)
print("loading complete")
images = images / 255.0 # Normalize data

train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)

label_encoder = LabelEncoder() # Numerical labels
numerical_labels = label_encoder.fit_transform(train_labels)
num_classes =19
unique_labels = np.unique(numerical_labels)
print("Unique labels:", unique_labels)

if np.any(unique_labels >= num_classes):
    raise ValueError("Invalid labels found. Ensure that labels are within the range [0, {}]".format(num_classes - 1))

#len(np.unique(labels))
one_hot_labels = to_categorical(numerical_labels, num_classes=num_classes)

test_numerical_labels = label_encoder.transform(test_labels)
test_one_hot_labels = to_categorical(test_numerical_labels, num_classes=num_classes)

train_labels = one_hot_labels
test_labels = test_one_hot_labels 
train_images = train_images.reshape(-1, 28, 28, 1) # Reshape data to fit the model
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
model.add(Dense(num_classes, activation='softmax'))  # Use num_classes instead of hardcoding 15

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train
history = model.fit(
    train_images, train_labels,
    epochs=60, 
    batch_size=32,
    validation_data=(test_images, test_labels)
)
# Save the trained model
model.save('my_digit_symbol_model.h5')

# Plot training & validation accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Saving the model
model.save('my_digit_symbol_model.keras')
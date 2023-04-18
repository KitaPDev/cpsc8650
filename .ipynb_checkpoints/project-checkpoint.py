import nibabel as nib
import pandas as pd
import os
import numpy as np
import random
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from scipy import ndimage

import tensorflow as tf
from tensorflow import keras
from keras.optimizers import Adam
from keras.layers import Conv3D, MaxPooling3D, GlobalAveragePooling3D, BatchNormalization, Dense, Dropout
from keras.metrics import Accuracy, MeanSquaredError, AUC, Recall, TruePositives, FalseNegatives, BinaryAccuracy


def load_nifti_image(file_path):
    """Read and load volume"""
    # Read file
    img = nib.load(file_path)
    # Get raw data
    img = img.get_fdata()
    return img


def normalize(img):
    """Normalize the img"""
    min = -1000
    max = 400
    img[img < min] = min
    img[img > max] = max
    img = (img - min) / (max - min)
    img = img.astype("float32")
    return img


def reshape(img):
    """Resize across z-axis"""
    # Set the desired depth
    img_desired_depth = 64
    img_desired_width = 128
    img_desired_height = 128
    # Get current depth
    current_depth = img.shape[-1]
    current_width = img.shape[0]
    current_height = img.shape[1]
    # Compute depth factor
    depth = current_depth / img_desired_depth
    width = current_width / img_desired_width
    height = current_height / img_desired_height
    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height
    # Rotate
    img = ndimage.rotate(img, 90, reshape=False)
    # Resize across z-axis
    img = ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=1)
    return img


def process_image(path):
    """Read and resize volume"""
    try:
        img = load_nifti_image(path)
        img = normalize(img)
        img = reshape(img)
        return img
    
    except Exception as e:
        print(path)
        print(repr(e))



features = pd.read_csv(os.path.join(os.getcwd(), "Label_file.csv"))
features.head()

file_names = os.listdir(os.path.join(os.getcwd(), "files"))
file_names[:2]

labels = np.array([])
for n in file_names:
    labels = np.append(labels, features[features["Filename"] == n.strip(".gz")]["Recognizable-Facial-Feature"])
labels[:2]

file_paths = [os.path.join(os.getcwd(), "files", f) for f in file_names]
file_paths[:2]

norm_images = np.array([process_image(p) for p in file_paths])

labels = np.array([1 if label == 'Yes' else 0 for label in labels])
labels[:5]

X_train, X_test, Y_train, Y_test = train_test_split(norm_images, labels, test_size=0.4, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(X_train, Y_train, test_size=0.4, random_state=42)

@tf.function
def rotate(volume):
    """Rotate the volume by a few degrees"""

    def scipy_rotate(volume):
        # define some rotation angles
        angles = [-20, -10, -5, 5, 10, 20]
        # pick angles at random
        angle = random.choice(angles)
        # rotate volume
        volume = ndimage.rotate(volume, angle, reshape=False)
        volume[volume < 0] = 0
        volume[volume > 1] = 1
        return volume

    aug_volume = tf.numpy_function(scipy_rotate, [volume], tf.float32)
    return aug_volume


def train_preprocessing(volume, label):
    """Process training data by rotating and adding a channel."""
    # Rotate volume
    volume = rotate(volume)
    volume = tf.expand_dims(volume, axis=3)
    return volume, label


def validation_preprocessing(volume, label):
    """Process validation data by only adding a channel."""
    volume = tf.expand_dims(volume, axis=3)
    return volume, label


#enter the train test split cell here and rename the variable in train loader and validation loader
train_loader = tf.data.Dataset.from_tensor_slices((x_train, y_train))
validation_loader = tf.data.Dataset.from_tensor_slices((tf.convert_to_tensor(x_val), tf.convert_to_tensor(y_val)))

batch_size = 2
# Augment the on the fly during training.
train_dataset = (
    train_loader.shuffle(len(x_train))
    .map(train_preprocessing)
    .batch(batch_size)
    .prefetch(2)
)
# Only rescale.
validation_dataset = (
    validation_loader.shuffle(len(x_val))
    .map(validation_preprocessing)
    .batch(batch_size)
    .prefetch(2)
)

def make_model(width=128, height=128, depth=64):
    """Build a 3D convolutional neural network model."""

    inputs = keras.Input((width, height, depth, 1))

    x = Conv3D(filters=64, kernel_size=3, activation="relu")(inputs)
    x = MaxPooling3D(pool_size=2)(x)
    x = BatchNormalization()(x)

    x = Conv3D(filters=64, kernel_size=3, activation="relu")(x)
    x = MaxPooling3D(pool_size=2)(x)
    x = BatchNormalization()(x)

    x = Conv3D(filters=128, kernel_size=3, activation="relu")(x)
    x = MaxPooling3D(pool_size=2)(x)
    x = BatchNormalization()(x)

    x = Conv3D(filters=256, kernel_size=3, activation="relu")(x)
    x = MaxPooling3D(pool_size=2)(x)
    x = BatchNormalization()(x)

    x = GlobalAveragePooling3D()(x)
    x = Dense(units=512, activation="relu")(x)
    x = Dropout(0.3)(x)

    outputs = Dense(units=1, activation="sigmoid")(x)

    # Define the model.
    model = keras.Model(inputs, outputs, name="3dcnn")
    return model


# Build model.
model = make_model(width=128, height=128, depth=64)
model.summary()

# Compile model.
initial_learning_rate = 0.0001
learning_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
)
model.compile(
    loss="binary_crossentropy",
    optimizer=Adam(learning_rate=learning_schedule),
    metrics=[Accuracy(), MeanSquaredError(), AUC(), Recall(), TruePositives(), FalseNegatives()],
)

# Define callbacks.
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    "3d_image_classification.h5", save_best_only=True
)
early_stopping_cb = tf.keras.callbacks.EarlyStopping(monitor="val_acc", patience=15)

# Train the model, doing validation at the end of each epoch
epochs = 5
model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=epochs,
    shuffle=True,
    verbose=2,
    callbacks=[checkpoint_cb, early_stopping_cb],
)

fig, ax = plt.subplots(1, 2, figsize=(20, 6))
ax = ax.ravel()

for i, metric in enumerate(["accuracy", "loss"]):
    ax[i].plot(model.history.history[metric])
    ax[i].plot(model.history.history["val_" + metric])
    ax[i].set_title("Model {}".format(metric))
    ax[i].set_xlabel("epochs")
    ax[i].set_ylabel(metric)
    ax[i].legend(["train", "validation"])

model.load_weights("3d_image_classification.h5")
new_list=[]
for i in range(len(X_test)):
    prediction = model.predict(np.expand_dims(X_test[i], axis=0))[0]
    scores = [1 - prediction[0], prediction[0]]
    new_list.append([prediction[0]])

test_list=[]
for i in Y_test:
    test_list.append([i])

m = BinaryAccuracy()
m.update_state(test_list, new_list)

model_accuracy=m.result().numpy()*100
model_accuracy

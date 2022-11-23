# Importing Required Libraries
import numpy as np
import os
import cv2
import tensorflow as tf
from tensorflow import keras
from PIL import Image
from sklearn.model_selection import train_test_split
np.random.seed(42)

# Importing support from wandb
import wandb
from wandb.keras import WandbCallback

#Set default hyperparameters
defaults = dict(
    dropout=0.2,
    hidden_layer_size=128,
    layer_1_size=16,
    layer_2_size=32,
    learn_rate=0.01,
    epochs=27,
    )
##TODO Connect yaml file for sweep configurations

#Initiate tracking
wandb.init() 

# #Initiate tracking
# wandb.init(config=defaults, project="gtsrb") 
# config = wandb.config
# print(config)

# Assigning Path for Dataset
ROOT_PATH = os.path.abspath(".")
data_dir = os.path.join(ROOT_PATH, 'input\gtsrb-german-traffic-sign')

# Resizing the images to 32x32x3
IMG_HEIGHT = 32
IMG_WIDTH = 32
channels = 3
# Setting Total Classes
NUM_CATEGORIES = 43

# Collecting the Training Data
image_data = []
image_labels = []

for i in range(NUM_CATEGORIES):
    path = data_dir + '\Train\\' + str(i)
    images = os.listdir(path)

    for img in images:
        try:
            image = cv2.imread(path + '\\' + img)
            image_fromarray = Image.fromarray(image, "RGB")
            resize_image = image_fromarray.resize((IMG_HEIGHT, IMG_WIDTH))
            image_data.append(np.array(resize_image))
            image_labels.append(i)
        except:
            print("Error in " + img)

# Changing the list to numpy array
image_data = np.array(image_data)
image_labels = np.array(image_labels)

# Shuffling the data
shuffle_indexes = np.arange(image_data.shape[0])
np.random.shuffle(shuffle_indexes)
image_data = image_data[shuffle_indexes]
image_labels = image_labels[shuffle_indexes]

# Splitting the data into train and validation set
X_train, X_val, y_train, y_val = train_test_split(image_data, image_labels, test_size=0.3, random_state=42, shuffle=True)

X_train = X_train/255 
X_val = X_val/255

print("X_train.shape", X_train.shape)
print("X_valid.shape", X_val.shape)
print("y_train.shape", y_train.shape)
print("y_valid.shape", y_val.shape)

# One hot encoding the labels
y_train = keras.utils.to_categorical(y_train, NUM_CATEGORIES)
y_val = keras.utils.to_categorical(y_val, NUM_CATEGORIES)

## Building the model with hyper parameters from mysweep.yaml

model = keras.models.Sequential([
    keras.layers.Conv2D(filters=wandb.config.layer_1_size, kernel_size=(3,3), activation='relu', input_shape=(IMG_HEIGHT,IMG_WIDTH,channels)),
    keras.layers.MaxPool2D(pool_size=(2, 2)),
    keras.layers.BatchNormalization(axis=-1),
    
    keras.layers.Conv2D(filters=wandb.config.layer_2_size, kernel_size=(3,3), activation='relu'),
    keras.layers.MaxPool2D(pool_size=(2, 2)),
    keras.layers.BatchNormalization(axis=-1),
    
    keras.layers.Flatten(),

    keras.layers.Dense(wandb.config.hidden_layer_size, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(rate=wandb.config.dropout),
    
    keras.layers.Dense(43, activation='softmax')
])

print(model.summary())

opt = tf.keras.optimizers.Adam(learning_rate=wandb.config.learn_rate, decay=wandb.config.decay)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

### Starting the training
history = model.fit(X_train, y_train, batch_size=wandb.config.batch_size, epochs=wandb.config.epochs, validation_data=(X_val, y_val), callbacks=[WandbCallback()])
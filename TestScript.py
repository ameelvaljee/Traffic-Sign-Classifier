# Importing Required Libraries
import numpy as np
import pandas as pd
import os
import cv2
from tensorflow import keras
from PIL import Image
from sklearn.metrics import accuracy_score

# Assigning Path for Dataset
ROOT_PATH = os.path.abspath(".")
data_dir = os.path.join(ROOT_PATH, 'input\gtsrb-german-traffic-sign')

# Resizing the images to 32x32x3
IMG_HEIGHT = 32
IMG_WIDTH = 32
NUM_CATEGORIES = 43

# Loading saved model
model = keras.models.load_model(ROOT_PATH +  '\\final_model\\files\model-best.h5')
print(model.summary())

# Loading the test data and running the predictions
test = pd.read_csv(data_dir + '\Test.csv')

labels = test["ClassId"].values
imgs = test["Path"].values

data =[]

for img in imgs:
    try:
        image = cv2.imread(data_dir + '\\' +img)
        image_fromarray = Image.fromarray(image, 'RGB')
        resize_image = image_fromarray.resize((IMG_HEIGHT, IMG_WIDTH))
        data.append(np.array(resize_image))
    except:
        print("Error in " + img)
X_test = np.array(data)
X_test = X_test/255

#one hot encoding the labels
Y_test = keras.utils.to_categorical(labels, NUM_CATEGORIES)

# Evaluating model on test data
val_loss_x=model.evaluate(X_test, Y_test, batch_size=32)

# Calculating and returning accuracy
print('Test data loss value: ' + str(val_loss_x[0]))
print('Test data accuracy: ' + str(val_loss_x[1]))
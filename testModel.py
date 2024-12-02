import sys
print(sys.version)

import torch
print(torch.__version__)

import tensorflow as tf
print(tf.__version__)
print("Is GPU available:", tf.config.list_physical_devices('GPU'))

import h5py
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

# import numpy as np
# import tensorflow as tf
# from keras.models import load_model
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.losses import SparseCategoricalCrossentropy
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, roc_auc_score
# from sklearn.ensemble import RandomForestClassifier
# import joblib
# from keras.saving import register_keras_serializable
#
# try:
#     # Load without compiling to fix the reduction issue
#     target_model = load_model('gtsrb_bottom_right_white_4_target_33.h5', compile=False)
#
#     # Recompile the model with the custom loss function
#     target_model.compile(optimizer=Adam(), metrics=['accuracy'])
#     model.summary()
#
#     # Save the model in the recommended Keras format
#     target_model.save('corrected_target_model.keras')
#     print("Model successfully recompiled and saved as 'corrected_target_model.keras'.")
# except Exception as e:
#     print("Error loading and recompiling the model:", str(e))
#     raise

from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
BATCH_SIZE = 32 # batch size used for optimization
# load the dataset
dataset = {}
with h5py.File((’%s/%s’ % (DATA_DIR, DATA_FILE)), ’r’) as hf:
    dataset[’X_test’] = np.array(hf.get(’X_test’))
    dataset[’Y_test’] = np.array(hf.get(’Y_test’))
X_test = np.array(dataset['X_test'], dtype=’float32’)
Y_test = np.array(dataset['Y_test'], dtype=’float32’)
# create the data generator
datagen = ImageDataGenerator()
test_generator = datagen.flow(X_test, Y_test, batch_size=BATCH_SIZE)
# load the infected model
model = load_model((’%s/%s’ % (MODEL_DIR, MODEL_FILENAME)))
model.summary()

import sys
print(sys.version)

import torch
print(torch.__version__)

import tensorflow as tf
print(tf.__version__)

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import joblib
from keras.saving import register_keras_serializable

try:
    # Load without compiling to fix the reduction issue
    target_model = load_model('gtsrb_bottom_right_white_4_target_33.h5', compile=False)

    # Recompile the model with the custom loss function
    target_model.compile(optimizer=Adam(), metrics=['accuracy'])
    model.summary()

    # Save the model in the recommended Keras format
    target_model.save('corrected_target_model.keras')
    print("Model successfully recompiled and saved as 'corrected_target_model.keras'.")
except Exception as e:
    print("Error loading and recompiling the model:", str(e))
    raise

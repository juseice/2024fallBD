# -*- coding: utf-8 -*-
"""BD.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1k43fY_C_WhPN5Ccj20T16dTtUT61C7eq
"""

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


import numpy as np
# os.pwd()
ROOT_DIR = ""
DATA_DIR = ROOT_DIR+'' # data folder
DATA_FILE = 'gtsrb_dataset_int.h5' # dataset file
MODEL_DIR = ROOT_DIR+'' # model directory
MODEL_FILENAME = 'gtsrb_bottom_right_white_4_target_33.h5' # model file
RESULT_DIR = '/results' # directory for storing results
# # image filename template for visualization results
# IMG_FILENAME_TEMPLATE = 'gtsrb_visualize_%s_label_%d.png'
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

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
BATCH_SIZE = 32 # batch size used for optimization
# load the dataset
dataset = {}
with h5py.File(('%s' % (DATA_FILE)), 'r') as hf:
    dataset['X_test'] = np.array(hf.get('X_test'))
    dataset['Y_test'] = np.array(hf.get('Y_test'))
X_test = np.array(dataset['X_test'], dtype='float32')
Y_test = np.array(dataset['Y_test'], dtype='float32')
# create the data generator
datagen = ImageDataGenerator()
test_generator = datagen.flow(X_test, Y_test, batch_size=BATCH_SIZE)
# load the infected model
model = load_model(('%s' % (MODEL_FILENAME)))
model.summary()

NUM_CLASSES = 43 # total number of classes in the model
Y_TARGET = 33 # (optional) infected target label used for prioritizing label scanning

# input size
IMG_ROWS = 32
IMG_COLS = 32
IMG_COLOR = 3
INPUT_SHAPE = (IMG_ROWS, IMG_COLS, IMG_COLOR)
# mask size
MASK_SHAPE = np.ceil(np.array(INPUT_SHAPE[0:2], dtype=float)).astype(int)
NB_SAMPLE = 1000 # number of samples in each mini-batch
MINI_BATCH = NB_SAMPLE // BATCH_SIZE # mini batch size used for early stop

# define the performer structure
import tensorflow.keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.metrics import categorical_accuracy
from decimal import Decimal
# trainining-related prameters
STEPS = 1000 # total optimization iterations
INIT_COST = 1e-3 # initial weight used for balancing two objectives
EPSILON = K.epsilon()# epsilon used in tanh
LR = 0.1 # learning rate
EARLY_STOP_THRESHOLD = 0.99 # early stop threshold
PATIENCE = 10 # patience
EARLY_STOP_PATIENCE = 2 * PATIENCE # early stop patience
# threshold of attack success rate for dynamically changing cost
ATTACK_SUCC_THRESHOLD = 0.99
EARLY_STOP = True # early stop flag
# min/max of mask
MASK_MIN = 0
MASK_MAX = 1
# min/max of raw pixel intensity
COLOR_MIN = 0
COLOR_MAX = 255

class Performer:
  def __init__(self, model):
    self.model = model
    mask = np.zeros(MASK_SHAPE)
    pattern = np.zeros(INPUT_SHAPE)
    mask = np.expand_dims(mask, axis=2)
    mask_tanh = np.zeros_like(mask)
    pattern_tanh = np.zeros_like(pattern)
    # prepare mask related tensors
    self.mask_tanh_tensor = K.variable(mask_tanh)
    mask_tensor_unrepeat = (K.tanh(self.mask_tanh_tensor) /
        (2 - EPSILON) + 0.5)
    self.mask_tensor = K.expand_dims(mask_tensor_unrepeat, axis=0)
    reverse_mask_tensor = (K.ones_like(self.mask_tensor) -
        self.mask_tensor)
    # prepare pattern related tensors
    self.pattern_tanh_tensor = K.variable(pattern_tanh)
    self.pattern_raw_tensor = (
        (K.tanh(self.pattern_tanh_tensor) / (2 - EPSILON) + 0.5) *
        255.0)
    # prepare the training model
    input_tensor = K.placeholder(model.input_shape)
    input_raw_tensor = input_tensor
    X_adv_raw_tensor = (
      reverse_mask_tensor * input_raw_tensor +
      self.mask_tensor * self.pattern_raw_tensor)
    X_adv_tensor = X_adv_raw_tensor
    output_tensor = model(X_adv_tensor)
    y_true_tensor = K.placeholder(model.output_shape)
    self.loss_acc = categorical_accuracy(output_tensor, y_true_tensor)
    self.loss_ce = categorical_crossentropy(output_tensor, y_true_tensor)
    # l1 regularization
    self.loss_reg = (K.sum(K.abs(self.mask_tensor)) / IMG_COLOR)
    cost = INIT_COST
    self.cost_tensor = K.variable(cost)
    self.loss = self.loss_ce + self.loss_reg * self.cost_tensor
    self.opt = tensorflow.keras.optimizers.Adam(learning_rate=LR, beta_1=0.5, beta_2=0.9)
    self.updates = self.opt.get_updates(
      params=[self.pattern_tanh_tensor, self.mask_tanh_tensor],
      loss=self.loss)
    self.train = K.function(
        [input_tensor, y_true_tensor],
        [self.loss_ce, self.loss_reg, self.loss, self.loss_acc],
        updates=self.updates)


  def reset_opt(self):
    K.set_value(self.opt.iterations, 0)
    for w in self.opt.weights:
      K.set_value(w, np.zeros(K.int_shape(w)))


  def reset_state(self, pattern_init, mask_init):
    print('resetting state')
    self.cost = INIT_COST
    K.set_value(self.cost_tensor, self.cost)
    # setting mask and pattern
    mask = np.array(mask_init)
    pattern = np.array(pattern_init)
    mask = np.clip(mask, MASK_MIN, MASK_MAX)
    pattern = np.clip(pattern, COLOR_MIN, COLOR_MAX)
    mask = np.expand_dims(mask, axis=2)
    # convert to tanh space
    mask_tanh = np.arctanh((mask - 0.5) * (2 - EPSILON))
    pattern_tanh = np.arctanh((pattern / 255.0 - 0.5) * (2 - EPSILON))
    print('mask_tanh', np.min(mask_tanh), np.max(mask_tanh))
    print('pattern_tanh', np.min(pattern_tanh), np.max(pattern_tanh))
    K.set_value(self.mask_tanh_tensor, mask_tanh)
    K.set_value(self.pattern_tanh_tensor, pattern_tanh)
    # resetting optimizer states
    self.reset_opt()


  def visualize(self, gen, y_target, pattern_init, mask_init):
    # since we use a single optimizer repeatedly, we need to reset
    # optimzier's internal states before running the optimization
    self.reset_state(pattern_init, mask_init)
    # best optimization results
    mask_best = None
    pattern_best = None
    reg_best = float('inf')
    # vectorized target
    Y_target = to_categorical([y_target] * BATCH_SIZE, NUM_CLASSES)
    # loop start
    for step in range(STEPS):
      # record loss for all mini-batches
      loss_ce_list = []
      loss_reg_list = []
      loss_list = []
      loss_acc_list = []
      for idx in range(MINI_BATCH):
        X_batch, _ = gen.next()
        if X_batch.shape[0] != Y_target.shape[0]:
          Y_target = to_categorical([y_target] * X_batch.shape[0], NUM_CLASSES)

      (loss_ce_value,
      loss_reg_value,
      loss_value,
      loss_acc_value) = self.train([X_batch, Y_target])

      loss_ce_list.extend(list(loss_ce_value.flatten()))
      loss_reg_list.extend(list(loss_reg_value.flatten()))
      loss_list.extend(list(loss_value.flatten()))
      loss_acc_list.extend(list(loss_acc_value.flatten()))
      avg_loss_ce = np.mean(loss_ce_list)
      avg_loss_reg = np.mean(loss_reg_list)
      avg_loss = np.mean(loss_list)
      avg_loss_acc = np.mean(loss_acc_list)

      # check to save best mask or not
      if avg_loss_acc >= ATTACK_SUCC_THRESHOLD and avg_loss_reg < reg_best:
        mask_best = K.eval(self.mask_tensor)
        mask_best = mask_best[0, ..., 0]
        pattern_best = K.eval(self.pattern_raw_tensor)
        reg_best = avg_loss_reg

      print('step: %3d, cost: %.2E, attack: %.3f, loss: %f, ce: %f, reg: %f, reg_best: %f' % (step, Decimal(self.cost), avg_loss_acc, avg_loss, avg_loss_ce, avg_loss_reg, reg_best))
    # save the final version
    if mask_best is None:
      mask_best = K.eval(self.mask_tensor)
      mask_best = mask_best[0, ..., 0]
      pattern_best = K.eval(self.pattern_raw_tensor)
    return pattern_best, mask_best


# create a performer
performer = Performer(model=model)

# define the y_target_list
y_target_list = list(range(NUM_CLASSES))
y_target_list.remove(Y_TARGET)
y_target_list = [Y_TARGET] + y_target_list

# start reverse engineering triggers
for y_target in y_target_list:
  print('processing label %d' % y_target)

  # initialize with random mask
  pattern = np.random.random(INPUT_SHAPE) * 255.0
  mask = np.random.random(MASK_SHAPE)

  # perform reverse training:
  pattern, mask = performer.visualize(gen=test_generator, y_target=y_target,
  pattern_init=pattern, mask_init=mask)

  print('pattern, shape: %s, min: %f, max: %f' %
  (str(pattern.shape), np.min(pattern), np.max(pattern)))
  print('mask, shape: %s, min: %f, max: %f' %
  (str(mask.shape), np.min(mask), np.max(mask)))
  print('mask norm of label %d: %f' %
  (y_target, np.sum(np.abs(mask))))
  save_pattern(pattern, mask, y_target)

# image filename template for visualization results
IMG_FILENAME_TEMPLATE = 'gtsrb_visualize_%s_label_%d.png'

def outlier_detection(l1_norm_list, idx_mapping):
  consistency_constant = 1.4826 # if normal distribution
  median = np.median(l1_norm_list)
  mad = consistency_constant * np.median(np.abs(l1_norm_list - median))
  min_mad = np.abs(np.min(l1_norm_list) - median) / mad

  print('median: %f, MAD: %f' % (median, mad))
  print('anomaly index: %f' % min_mad)

  flag_list = []
  for y_label in idx_mapping:
    if l1_norm_list[idx_mapping[y_label]] > median:
      continue
    if np.abs(l1_norm_list[idx_mapping[y_label]] - median) / mad > 2:
      flag_list.append((y_label, l1_norm_list[idx_mapping[y_label]]))
  if len(flag_list) > 0:
    flag_list = sorted(flag_list, key=lambda x: x[1])

  print('flagged label list: %s' %
    ', '.join(['%d: %2f' % (y_label, l_norm)
        for y_label, l_norm in flag_list]))


mask_flatten = []
idx_mapping = {}

for y_label in range(NUM_CLASSES):
  mask_filename = IMG_FILENAME_TEMPLATE % ('mask', y_label)
  if os.path.isfile('%s/%s' % (RESULT_DIR, mask_filename)):
    img = image.load_img(
        '%s/%s' % (RESULT_DIR, mask_filename),
        color_mode='grayscale',
        target_size=INPUT_SHAPE)
    mask = image.img_to_array(img)
    mask /= 255
    mask = mask[:, :, 0]
    mask_flatten.append(mask.flatten())
    idx_mapping[y_label] = len(mask_flatten) - 1

l1_norm_list = [np.sum(np.abs(m)) for m in mask_flatten]
print('%d labels found' % len(l1_norm_list))
outlier_detection(l1_norm_list, idx_mapping)

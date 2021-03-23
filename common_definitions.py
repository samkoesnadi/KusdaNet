# common imports
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from math import ceil
import datetime

# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

tf.random.set_seed(0)
np.random.seed(0)

# # On TPUs, use 'mixed_bfloat16' instead
# from tensorflow.keras.mixed_precision import experimental as mixed_precision
# policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
# mixed_precision.set_policy(policy)

# common global variables
IMAGE_INPUT_SIZE = 224  # this is because of Xception
NUM_CLASSES = 14
LOAD_WEIGHT_BOOL = True
DROPOUT_N = 0.3
KERNEL_INITIALIZER = tf.keras.initializers.he_normal()
GLOBAL_ACTIVATION = tf.nn.leaky_relu

# network model related variable
USE_SVM = False
USE_PATIENT_DATA = False
USE_WN = False
USE_CONV1D = False
USE_DOM_ADAP_NET = True
# LAST_ACTIVATION = tf.nn.relu6
LAST_ACTIVATION = "tanh"
TRAIN_CHEXPERT = True  # important if false then, it is trained on chestxray14
PAT_DATA_SIZE = 4
NUM_FEATURES = 2048 + (PAT_DATA_SIZE if USE_PATIENT_DATA else 0)

# train settings
# USE_CLASS_WEIGHT = False
USE_SPARSITY_NORM = False
USE_AUGMENTATION = False
USE_GAN = True
USE_CLR = False
USE_EARLY_STOPPING = False
MODELCKP_BEST_ONLY = not USE_DOM_ADAP_NET
USE_DROPOUT_PAT_DATA = True
BUFFER_SIZE = 16000
BATCH_SIZE = 32  # 32 is optimal
# BUFFER_SIZE = 1600
# BATCH_SIZE = 16  # 32 is optimal
MAX_EPOCHS = 20
LEARNING_RATE = 1e-4
# ACTIVIY_REGULARIZER_VAL = 1e-3  # TODO: check this value out

# for training domAdap
LAMBDA_ADV = .001  # .001
DISC_LEARNING_RATE = 1e-4
USE_SOFT_LABEL_SMOOTHING = False
USE_NOISY_LABEL = False
SL_LOWERBOUND = [0.0, 0.1]
SL_UPPERBOUND = [0.9, 1.0]
NOISY_LABEL_PERCENTAGE = ceil(5./100. * BATCH_SIZE)

# eval settings
EVAL_CHEXPERT = True  # important if false then, it is trained on chestxray14
AUC_INTERP_TOGGLE = False  # activate AUC interp

# target dataset FeLoss conf
LABELS_COUPLE_INDEX = [
    [2, 5, 6, 8, 10, 7, 9],  # CheXpert
    [1, 9, 8, 0, 2, 6, 7]
]

# for manifold learning
MODEL_SVM_PATH = "/mnt/7E8EEE0F8EEDBFAF/project/bachelorThesis/records/all_trainings/20200126-034328/checkpoints/model_weights.04-0.86.hdf5"
MODEL_CHEXPERT_PATH = "/mnt/7E8EEE0F8EEDBFAF/project/bachelorThesis/records/all_trainings/20200326-093110/checkpoints/model_weights.04-0.87.hdf5"
MODEL_CHESTXRAY_PATH = "/mnt/7E8EEE0F8EEDBFAF/project/bachelorThesis/records/all_trainings/20200326-092905/checkpoints/model_weights.04-0.86.hdf5"

# evaluation matter
TRAIN_FIVE_CATS_INDEX = [2, 5, 6, 8, 10] if TRAIN_CHEXPERT else [1, 9, 8, 0, 2]
EVAL_FIVE_CATS_INDEX = [2, 5, 6, 8, 10] if EVAL_CHEXPERT else [1, 9, 8, 0, 2]

# SVM
SVM_KERNEL_REGULARIZER = 0.5  # 0.5 is according to the paper

# AUC interpolation
INTERP_NUM_STEPS = 10000

TENSORBOARD_LOGDIR = "./logs/kusdaNet/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# callbacks hyperparameter
# clr
CLR_BASELR = 1e-4
CLR_MAXLR = 1e-3
CLR_PATIENCE = 2

MODELCKP_PATH = "./checkpoints/model_weights.{epoch:02d}-{val_auc:.2f}.hdf5"  # do not change the format of basename

SAVED_MODEL_PATH = './weights/model.h5'

# for validation
THRESHOLD_SIGMOID = 0.5
SAMPLE_FILENAME = "./sample/00002032_012.png"
SAMPLE_PATIENT_DATA = [0.5, 55 / 100, 0, 0.5]

# for evaluation
ROC_RESULTS_PATH = "./report/results/ROC_%s.png"
AUC_RESULTS_PATH = "./report/results/AUC.txt"

# dataset common
VALID_RATIO = 10 / 100

# cheXpert dataset
CHEXPERT_TRAIN_TARGET_TFRECORD_PATH = './cheXpert_datasets/CheXpert_train.tfrecord'
CHEXPERT_VALID_TARGET_TFRECORD_PATH = './cheXpert_datasets/CheXpert_valid.tfrecord'
CHEXPERT_TEST_TARGET_TFRECORD_PATH = './cheXpert_datasets/CheXpert_test.tfrecord'
CHEXPERT_DATASET_PATH = "../datasets"

CHEXPERT_TRAIN_N = 201073
CHEXPERT_VAL_N = 22341
CHEXPERT_TEST_N = 234

CHEXPERT_LABELS_KEY = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 'Lung Lesion',
                       'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
                       'Pleural Other', 'Fracture', 'Support Devices']

# chestXray-14 dataset
CHESTXRAY_TRAIN_TARGET_TFRECORD_PATH = 'cheXray14_datasets/CheXray14_train.tfrecord'
CHESTXRAY_VALID_TARGET_TFRECORD_PATH = 'cheXray14_datasets/CheXray14_valid.tfrecord'
CHESTXRAY_TEST_TARGET_TFRECORD_PATH = 'cheXray14_datasets/CheXray14_test.tfrecord'
CHESTXRAY_DATASET_PATH = "../datasets/chestXray14/images"

CHESTXRAY_TRAIN_N = 77872
CHESTXRAY_VAL_N = 8652
CHESTXRAY_TEST_N = 25596

CHESTXRAY_LABELS_KEY = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", "Pneumonia",
                        "Pneumothorax", "Consolidation", "Edema", "Emphysema", "Fibrosis", "Pleural_Thickening",
                        "Hernia"]

# conclusion of dataset
if TRAIN_CHEXPERT:
    TRAIN_TARGET_TFRECORD_PATH = CHEXPERT_TRAIN_TARGET_TFRECORD_PATH
    VALID_TARGET_TFRECORD_PATH = CHEXPERT_VALID_TARGET_TFRECORD_PATH
    TEST_TARGET_TFRECORD_PATH = CHEXPERT_TEST_TARGET_TFRECORD_PATH
    DATASET_PATH = CHEXPERT_DATASET_PATH
    TRAIN_N = CHEXPERT_TRAIN_N
    VAL_N = CHEXPERT_VAL_N
    LABELS_KEY = CHEXPERT_LABELS_KEY
else:
    TRAIN_TARGET_TFRECORD_PATH = CHESTXRAY_TRAIN_TARGET_TFRECORD_PATH
    VALID_TARGET_TFRECORD_PATH = CHESTXRAY_VALID_TARGET_TFRECORD_PATH
    TEST_TARGET_TFRECORD_PATH = CHESTXRAY_TEST_TARGET_TFRECORD_PATH
    DATASET_PATH = CHESTXRAY_DATASET_PATH
    TRAIN_N = CHESTXRAY_TRAIN_N
    VAL_N = CHESTXRAY_VAL_N
    LABELS_KEY = CHESTXRAY_LABELS_KEY

if EVAL_CHEXPERT:
    TEST_N = CHEXPERT_TEST_N
else:
    TEST_N = CHESTXRAY_TEST_N

    # statistics training
pos = dict()
pos[0] = 20155
pos[1] = 20857
pos[2] = 31635
pos[3] = 100087
pos[4] = 9603
pos[5] = 58583
pos[6] = 38255
pos[7] = 22401
pos[8] = 60412
pos[9] = 20324
pos[10] = 88072
pos[11] = 5536
pos[12] = 8666
pos[13] = 105344

neg = dict()
neg[0] = 180918
neg[1] = 180216
neg[2] = 169438
neg[3] = 100986
neg[4] = 191470
neg[5] = 142490
neg[6] = 162818
neg[7] = 178672
neg[8] = 140661
neg[9] = 180749
neg[10] = 113001
neg[11] = 195537
neg[12] = 192407
neg[13] = 95729

# calculation
CHEXPERT_CLASS_WEIGHT = \
    np.array([[0.55570203, 4.98816671],
              [0.55786667, 4.82027617],
              [0.59335273, 3.17801486],
              [0.99554889, 1.00449109],
              [0.52507704, 10.46928043],
              [0.70556881, 1.71613779],
              [0.6174778, 2.62806169],
              [0.56268749, 4.48803625],
              [0.71474325, 1.66418096],
              [0.55622161, 4.94668864],
              [0.88969567, 1.14152625],
              [0.51415589, 18.16049494],
              [0.52251997, 11.60125779],
              [1.05021989, 0.9543638]])

# K_SN cannot be 0.
if IMAGE_INPUT_SIZE == 320:
    K_SN = 101313.55625  # 320x320 IMAGE SIZE
elif IMAGE_INPUT_SIZE == 224:
    K_SN = 49721.094
else:
    K_SN = 1.

FEATURES_NP_FILE_1 = "../records/chextpert_train_input_features"
FEATURES_NP_FILE_2 = "../records/chestray14_train_input_features"

# kwd and manifold_learning
if TRAIN_CHEXPERT:
    FEATURES_NP_FILE = FEATURES_NP_FILE_1
else:
    FEATURES_NP_FILE = FEATURES_NP_FILE_2

# # normalize RATIO_LOSSES
# _total_ratio = RATIO_LOSSES[1] + RATIO_LOSSES[0]
# RATIO_LOSSES[1] /= _total_ratio
# RATIO_LOSSES[0] /= _total_ratio

# for ADDA
BASELINE_WEIGHT_PATH = "./baseline_weight/model_weights.{epoch:02d}-{val_auc:.2f}.hdf5"

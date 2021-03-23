from sklearn.manifold import Isomap, TSNE
from common_definitions import *
import time
from utils.visualization import *
from datasets.cheXpert_dataset import read_dataset
from models.multi_label import model_binaryXE_mid
from models.multi_class import model_MC_SVM
from utils.utils import _np_to_binary
import sklearn.metrics
from models.gan import *

PRINT_PREDICTION = False
FEATURE_LAYER_NAME = 1
# FEATURE_LAYER_NAME = "features"

if __name__ == "__main__":
    if USE_SVM:
        model = model_MC_SVM(with_feature=True)
    elif USE_DOM_ADAP_NET:
        model = GANModel()
        # to initiate the graph
        model(tf.zeros((1, IMAGE_INPUT_SIZE, IMAGE_INPUT_SIZE, 1)))
    else:
        model = model_binaryXE_mid(use_patient_data=USE_PATIENT_DATA)

    if LOAD_WEIGHT_BOOL:
        target_model_weight, _ = get_max_acc_weight(MODELCKP_PATH)
        if target_model_weight:  # if weight is Found
            model.load_weights(target_model_weight)
        else:
            print("[Load weight] No weight is found")

    test_dataset = read_dataset(
        CHEXPERT_TEST_TARGET_TFRECORD_PATH if EVAL_CHEXPERT else CHESTXRAY_TEST_TARGET_TFRECORD_PATH,
        CHEXPERT_DATASET_PATH if EVAL_CHEXPERT else CHESTXRAY_DATASET_PATH, evaluation_mode=True, use_patient_data=USE_PATIENT_DATA)

    _test_n = CHEXPERT_TEST_N  # TODO

    _color_label = None
    _feature_nps = []
    for i_test, (input, label) in tqdm(enumerate(test_dataset.take(_test_n))):
        predictions = model.predict(input) if not USE_DOM_ADAP_NET else model.call_w_features(input)

        label = (predictions[0][:, TRAIN_FIVE_CATS_INDEX] >= 0.3).astype(np.float32) if PRINT_PREDICTION else label.numpy()
        # feature_vectors = tf.reduce_mean(predictions[FEATURE_LAYER_NAME], axis=[1,2]).numpy()
        feature_vectors = predictions[FEATURE_LAYER_NAME].numpy()

        # filter zeros
        _i_zeros = np.argwhere(np.array(list(map(_np_to_binary, label))) != 0)[:, 0]
        label = label[_i_zeros]
        feature_vectors = feature_vectors[_i_zeros]

        labels = 1 - sklearn.metrics.pairwise.cosine_similarity(np.eye(5), label)

        if _color_label is None:
            _color_label = labels
        else:
            _color_label = np.concatenate((_color_label, labels), axis=-1)
        _feature_nps.extend(feature_vectors)

    # convert to np arrays
    _feature_nps = np.array(_feature_nps)

    for _i_c, _col_lab in enumerate(_color_label):
        start_time = time.time()
        embedding = TSNE(n_components=2, init='pca', random_state=0, verbose=True)
        X_embedded = embedding.fit_transform(_feature_nps)
        print("time spent for manifold learning:", time.time() - start_time)

        # sketch it
        if EVAL_CHEXPERT:
            _scatter_plt = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=_col_lab, cmap=plt.cm.Spectral)
        else:
            _scatter_plt = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=_col_lab, cmap=plt.cm.Spectral, s=5)

        plt.colorbar(_scatter_plt)
        plt.axis('tight')

        get_and_mkdir("report/results/manifold_learning.png")
        plt.savefig("report/results/manifold_learning_{}.png".format(_i_c), bbox_inches="tight")
        plt.clf()
"""
Train normal model with binary XE as loss function
"""
from common_definitions import *
from datasets.cheXpert_dataset import read_dataset
from utils.utils import *
from utils.visualization import *
from models.multi_label import *

if __name__ == "__main__":
	model = model_binaryXE()

	if LOAD_WEIGHT_BOOL:
		target_model_weight, _ = get_max_acc_weight(MODELCKP_PATH)
		if target_model_weight:  # if weight is Found
			model.load_weights(target_model_weight)
		else:
			print("[Load weight] No weight is found")


	# get the dataset
	# train_dataset = read_dataset(CHEXPERT_TRAIN_TARGET_TFRECORD_PATH, CHEXPERT_DATASET_PATH)
	# val_dataset = read_dataset(CHEXPERT_VALID_TARGET_TFRECORD_PATH, CHEXPERT_DATASET_PATH)
	test_dataset = read_dataset(CHEXPERT_TEST_TARGET_TFRECORD_PATH, CHEXPERT_DATASET_PATH)

	test_labels = []
	# get the ground truth labels
	for _, test_label in test_dataset:
		test_labels.extend(test_label)
	test_labels = np.array(test_labels)


	# Evaluate the model on the test data using `evaluate`
	results = model.predict(test_dataset,
	                         steps=ceil(CHEXPERT_TEST_N / BATCH_SIZE),
	                         verbose=1)

	print("F1: ", np.mean(f1(test_labels, results).numpy()))

	plot_roc(test_labels, results)
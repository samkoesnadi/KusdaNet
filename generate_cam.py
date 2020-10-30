"""
Train normal model with binary XE as loss function
"""
from common_definitions import *
from datasets.cheXpert_dataset import *
from utils.utils import *
from utils.visualization import *
from models.multi_label import *
import skimage
import skimage.color
from models.gan import *
from models.multi_class import *


target_filename = "./sample/00002032_006.png"
target_filename = "/mnt/7E8EEE0F8EEDBFAF/project/bachelorThesis/records/bachelorThesis/predictions/cardiomegaly.png"
if __name__ == "__main__":
	if USE_SVM:
		model = model_MC_SVM()
	elif USE_DOM_ADAP_NET:
		model = GANModel()
		# to initiate the graph
		model(tf.zeros((1, IMAGE_INPUT_SIZE, IMAGE_INPUT_SIZE, 1)))
	else:
		model = model_binaryXE(use_patient_data=USE_PATIENT_DATA)

	if LOAD_WEIGHT_BOOL:
		target_model_weight, _ = get_max_acc_weight(MODELCKP_PATH)
		if target_model_weight:  # if weight is Found
			model.load_weights(target_model_weight)
		else:
			print("[Load weight] No weight is found")

	model = tf.keras.Model(inputs=model.input_layer, outputs=model.call_w_everything(model.input_layer))

	# the data
	_image = read_image_and_preprocess(target_filename, use_sn=False, use_preprocess_img=True)
	image_ori = skimage.color.gray2rgb(read_image_and_preprocess(target_filename, use_sn=False, use_preprocess_img=False))

	image = np.reshape(_image, (-1, IMAGE_INPUT_SIZE, IMAGE_INPUT_SIZE, 1))

	patient_data = np.zeros((1,4))
	import time
	start_time = time.time()
	if USE_PATIENT_DATA:
		prediction = model.predict({"input_img": image, "input_semantic": patient_data})[0]
	else:
		prediction = np.squeeze(model.predict(image)[0])
	print("Time spent", time.time()-start_time)

	gradcampps = Xception_gradcampp(model, image, patient_data=patient_data)
	# gradcampps = np.zeros((NUM_CLASSES, IMAGE_INPUT_SIZE, IMAGE_INPUT_SIZE, 3))

	results = np.zeros((NUM_CLASSES, IMAGE_INPUT_SIZE, IMAGE_INPUT_SIZE, 3))

	for i_g, gradcampp in enumerate(gradcampps):
		gradcampp = convert_to_RGB(gradcampp)

		# show 2 images side by side
		fig = plt.figure(figsize=(12.8, 9.6))
		ax1 = fig.add_subplot(1, 2, 1)
		ax1.imshow(image_ori)
		plt.axis("off")
		ax2 = fig.add_subplot(1, 2, 2)
		ax2.imshow(gradcampp)
		plt.axis("off")
		plt.title(LABELS_KEY[i_g] + "({:.2f})".format(prediction[i_g]))
		plt.savefig(f"gradcam_results/gradcam_{i_g+1}.png", bbox_inches='tight', dpi=100)
		plt.close(fig)

		result = .7 * image_ori + .3 * gradcampp
		results[i_g] = result


	max_row_n = ceil(math.sqrt(NUM_CLASSES + 1))
	f, axarr = plt.subplots(max_row_n, max_row_n)

	# show input / ori image
	axarr[0, 0].imshow(image_ori)
	axarr[0, 0].set_title("Input")
	axarr[0, 0].axis('off')

	for i_r, result in enumerate(results, 1):
		axarr[i_r // max_row_n, i_r % max_row_n].imshow(result)
		axarr[i_r // max_row_n, i_r % max_row_n].set_title(LABELS_KEY[i_r - 1] + "({:.2f})".format(prediction[i_r - 1]))
		axarr[i_r // max_row_n, i_r % max_row_n].axis('off')

	plt.grid(b=None)
	plt.axis('off')
	plt.show()


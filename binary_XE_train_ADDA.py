"""
Train normal model with ADDA
1. The baseline weight is available as baseline
2. Only the target CNN is updated
3. Target CNN weight is stored
"""
from tensorflow.python.keras.callbacks import configure_callbacks
from tqdm import tqdm

from _callbacks import get_callbacks
from datasets.cheXpert_dataset import read_dataset
from models.discriminator import make_ADDA_discriminator_model
from utils._auc import AUC
from utils.visualization import *
from models.gan import *

# global local vars
TARGET_DATASET_FILENAME = CHESTXRAY_TRAIN_TARGET_TFRECORD_PATH
TARGET_DATASET_PATH = CHESTXRAY_DATASET_PATH

if __name__ == "__main__":
    source_model = GANModel()
    target_model = GANModel()
    discriminator = make_ADDA_discriminator_model()

    # to initiate the graph
    source_model(tf.zeros((1, IMAGE_INPUT_SIZE, IMAGE_INPUT_SIZE, 1)))
    target_model(tf.zeros((1, IMAGE_INPUT_SIZE, IMAGE_INPUT_SIZE, 1)))

    # get the dataset
    train_dataset = read_dataset(TRAIN_TARGET_TFRECORD_PATH, DATASET_PATH,
                                 use_augmentation=USE_AUGMENTATION,
                                 use_patient_data=USE_PATIENT_DATA,
                                 use_feature_loss=False,
                                 use_preprocess_img=True)

    val_dataset = read_dataset(VALID_TARGET_TFRECORD_PATH, DATASET_PATH,
                               use_patient_data=USE_PATIENT_DATA,
                               use_feature_loss=False,
                               use_preprocess_img=True)
    test_dataset = read_dataset(TEST_TARGET_TFRECORD_PATH, DATASET_PATH,
                                use_patient_data=USE_PATIENT_DATA,
                                use_feature_loss=False,
                                use_preprocess_img=True)

    train_target_dataset = read_dataset(TARGET_DATASET_FILENAME, TARGET_DATASET_PATH,
                                        use_augmentation=False,
                                        use_patient_data=USE_PATIENT_DATA,
                                        use_feature_loss=False,
                                        use_preprocess_img=True,
                                        repeat=True)

    # losses, optimizer, metrics
    _XEloss = tf.keras.losses.BinaryCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.AUTO)

    # optimizer
    _optimizer = tf.keras.optimizers.Adam(LEARNING_RATE, amsgrad=True)
    _optimizer_disc = tf.keras.optimizers.Adam(DISC_LEARNING_RATE, amsgrad=True)

    # _metric = AUC(name="auc", multi_label=True, num_classes=NUM_CLASSES)  # give recall for metric it is more accurate
    _metric = tf.keras.metrics.AUC(name="auc")  # give recall for metric it is more accurate
    _callbacks = get_callbacks()

    # build CallbackList
    _callbackList = configure_callbacks(_callbacks,
                                        target_model,
                                        do_validation=True,
                                        epochs=MAX_EPOCHS,
                                        mode=tf.estimator.ModeKeys.EVAL,
                                        verbose=0)

    # retrieve checkpoint
    init_epoch = 0
    source_model_weight, _ = get_max_acc_weight(BASELINE_WEIGHT_PATH)
    assert source_model_weight
    source_model.load_weights(source_model_weight)
    target_model.load_weights(source_model_weight)

    # set all the parameters
    model_params = {
        "optimizer": _optimizer,
        "loss": _XEloss,
        "metrics": [_metric]
    }

    target_model.compile(**model_params)  # compile model

    fit_params = {
        "epochs": MAX_EPOCHS,
        "validation_data": val_dataset,
        # "initial_epoch": 0,
        # "steps_per_epoch":2,
        "callbacks": _callbacks,
        "verbose": 1
    }

    # save checkpoints
    checkpoint_dir = './checkpoints/disc'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(
        optimizer=_optimizer,
        optimizer_disc=_optimizer_disc,
        discriminator_optimizer=_optimizer_disc,
        discriminator=discriminator)


    class TrainWorker:
        def __init__(self, metric, _target_dataset, lambda_adv=0.001):
            self.metric = metric
            self.lambda_adv = lambda_adv
            self._target_dataset = iter(_target_dataset)
            self._eval_indices = tf.constant([1, 9, 8, 0, 2])

            self._keras_eps = tf.keras.backend.epsilon()

        def soft_entropy(self, y_true_range: list, y_pred):
            y_true = tf.random.uniform(tf.shape(y_pred), minval=y_true_range[0], maxval=y_true_range[1])

            return y_true * tf.math.log(y_pred + self._keras_eps) + (1. - y_true) * tf.math.log(
                    1. - y_pred + self._keras_eps)

                # Notice the use of `tf.function`
        # This annotation causes the function to be "compiled".
        @tf.function
        def gan_train_step(self, source_image_batch, source_label_batch):
            target_data = next(self._target_dataset)
            target_image_batch = target_data[0]
            target_label_batch = target_data[1]

            with tf.GradientTape(persistent=True) as g:
                source_predictions = source_model.call_w_features(source_image_batch, training=True)
                target_predictions = target_model.call_w_features(target_image_batch, training=True)

                # stop gradient for the output label
                source_disc_output = discriminator(source_predictions[1], training=True)
                target_disc_output = discriminator(target_predictions[1], training=True)

                # calculate xe loss
                source_xe_loss = _XEloss(source_label_batch, source_predictions[0])
                target_xe_loss = _XEloss(target_label_batch, target_predictions[0])

                gen_loss = tf.math.log(source_disc_output + self._keras_eps) + \
                            tf.math.log(1 - target_disc_output + self._keras_eps)  # BATCH * NUM_CLASSES
                disc_loss = tf.math.log(target_disc_output + self._keras_eps) + \
                            tf.math.log(1 - source_disc_output + self._keras_eps)

                # reduce mean gen and disc
                gen_loss = -tf.reduce_mean(gen_loss)
                disc_loss = -tf.reduce_mean(disc_loss)

            gradients_of_model = g.gradient(gen_loss, target_model.trainable_variables)
            gradients_of_discriminator = g.gradient(disc_loss, discriminator.trainable_variables)
            avg_grad_model = (tf.reduce_mean(tf.concat([tf.reshape(tf.math.abs(grad), [-1]) for grad in gradients_of_model], axis=-1)))
            avg_grad_disc = (tf.reduce_mean(tf.concat([tf.reshape(tf.math.abs(grad), [-1]) for grad in gradients_of_discriminator], axis=-1)))

            del g  # delete the persistent gradientTape

            _optimizer_disc.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
            _optimizer.apply_gradients(zip(gradients_of_model, target_model.trainable_variables))

            # calculate metrics
            self.metric.update_state(source_label_batch, source_predictions[0])

            return source_xe_loss, gen_loss, disc_loss, target_xe_loss, avg_grad_model, avg_grad_disc

    # initiate worker
    trainWorker = TrainWorker(_metric, _target_dataset=train_target_dataset, lambda_adv=LAMBDA_ADV)

    # # load disc and optimizer checkpoints
    # checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    # training loop
    _callbackList.on_train_begin()

    # var for save checkpoint
    _global_auc = 0.
    num_losses = 7
    losses = [tf.keras.metrics.Mean() for _ in range(num_losses)]

    for epoch in range(init_epoch, fit_params["epochs"]):
        print("Epoch %d/%d" % (epoch + 1, fit_params["epochs"]))
        _callbackList.on_epoch_begin(epoch)  # on epoch start

        # reset losses mean
        [loss.reset_states() for loss in losses]

        # g = trainWorker.gan_train_step if USE_DOM_ADAP_NET and (epoch % 2) else trainWorker.xe_train_step
        g = trainWorker.gan_train_step

        # if USE_AUGMENTATION:
        #     if epoch % 2:
        #         train_dataset = noaug_train_dataset
        #     else:
        #         train_dataset = aug_train_dataset
        # else:
        #     train_dataset = noaug_train_dataset

        with tqdm(total=math.ceil(TRAIN_N / BATCH_SIZE),
                  postfix=[dict()]) as t:
            for i_batch, (source_image_batch, source_label_batch) in enumerate(train_dataset):
                _batch_size = tf.shape(source_image_batch)[0].numpy()
                _callbackList.on_batch_begin(i_batch, {"size": _batch_size})  # on batch begin

                _losses = g(source_image_batch, source_label_batch)
                _auc = trainWorker.metric.result().numpy()

                # update loss
                [losses[i].update_state(_losses[i]) for i in range(num_losses - 1)]
                losses[num_losses - 1].update_state(_auc)

                # update tqdm
                # t.postfix[0]["_g"] = update_gen
                t.postfix[0]["xe_l"] = losses[0].result().numpy()
                t.postfix[0]["g_l"] = losses[1].result().numpy()
                t.postfix[0]["d_l"] = losses[2].result().numpy()
                t.postfix[0]["txe_l"] = losses[3].result().numpy()
                t.postfix[0]["avg_g_m"] = losses[4].result().numpy()
                t.postfix[0]["avg_g_d"] = losses[5].result().numpy()
                t.postfix[0]["AUC"] = losses[6].result().numpy()

                t.update()

                _callbackList.on_batch_end(i_batch, {"loss": losses[0].result()})  # on batch end

        # epoch_end
        print()
        print("Validating...")
        results = target_model.evaluate(val_dataset, callbacks=_callbacks)

        _callbackList.on_epoch_end(epoch, {"loss": losses[0].result(),
                                           "gen_loss": losses[1].result(),
                                           "disc_loss": losses[2].result(),
                                           "txe_loss": losses[3].result(),
                                           "avg_grad_m": losses[4].result(),
                                           "avg_grad_d": losses[5].result(),
                                           "auc": _auc,
                                           "val_loss": results[0],
                                           "val_auc": results[1],
                                           "lr": target_model.optimizer.lr})  # on epoch end

        # reset states
        trainWorker.metric.reset_states()

        # save checkpoint
        if USE_DOM_ADAP_NET:
            if _auc > _global_auc:
                _global_auc = _auc
                checkpoint.save(file_prefix=checkpoint_prefix)

    _callbackList.on_train_end()

    # Evaluate the model on the test data using `evaluate`
    results = target_model.evaluate(test_dataset)
    print('test loss, test f1, test auc:', results)

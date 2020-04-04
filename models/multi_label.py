"""
Normal model with binary XE as loss function
"""
from common_definitions import *
import tensorflow_addons as tfa


def raw_model_binaryXE(use_patient_data=False):
    input_layer = tf.keras.layers.Input(shape=(IMAGE_INPUT_SIZE, IMAGE_INPUT_SIZE, 1), name="input_img")
    image_section_model = tf.keras.applications.xception.Xception(include_top=False, weights=None, pooling="avg",
                                                                  input_tensor=input_layer)
    image_feature_vectors = image_section_model.output

    if use_patient_data:
        # process semantic
        input_semantic = tf.keras.layers.Input(shape=4, name="input_semantic")
        int_semantic = tf.keras.layers.Dropout(0.2)(input_semantic) if USE_DROPOUT_PAT_DATA else input_semantic

        feature_vectors_1 = tf.keras.layers.Concatenate()([image_feature_vectors, int_semantic])

        # Apply Batch Normalization to convert the range ro mean 0 and std 1
        feature_vectors_1 = tf.keras.layers.BatchNormalization()(feature_vectors_1)

        if USE_PATIENT_DATA_OPT_LAYER:
            feature_vectors = tf.keras.layers.Dense(2048, activation=tf.nn.leaky_relu,
                                                    kernel_initializer=KERNEL_INITIALIZER)(feature_vectors_1)
            feature_vectors = tf.keras.layers.BatchNormalization()(feature_vectors)
        else:
            feature_vectors = feature_vectors_1
    else:
        feature_vectors = image_feature_vectors

    image_section_layer = feature_vectors

    # add dropout if needed
    if DROPOUT_N != 0.:
        image_section_layer = tf.keras.layers.Dropout(DROPOUT_N)(image_section_layer)

    output_layer = tf.keras.layers.Dense(NUM_CLASSES)(image_section_layer)
    output_layer = tf.keras.layers.Activation('sigmoid', dtype='float32', name='predictions')(output_layer)

    # output_layer_object = tf.keras.layers.Dense(NUM_CLASSES,
    #                                             kernel_regularizer=tf.keras.regularizers.L1L2(l1=1e-3, l2=1e-3),
    #                                             bias_regularizer=tf.keras.regularizers.L1L2(l1=1e-2, l2=1e-2))
    # weight_normalized = tfa.layers.WeightNormalization(output_layer_object, dtype=tf.float32)(
    #     image_section_layer)
    # output_layer = tf.keras.layers.Activation('sigmoid', dtype='float32', name='predictions')(weight_normalized)

    if use_patient_data:
        return input_layer, input_semantic, output_layer, image_feature_vectors
    else:
        return input_layer, output_layer, image_feature_vectors


def model_binaryXE(use_patient_data=False):
    if use_patient_data:
        input_layer, input_semantic, output_layer, _ = raw_model_binaryXE(use_patient_data)

        model = tf.keras.Model(inputs=[input_layer, input_semantic], outputs=[output_layer])
    else:
        input_layer, output_layer, _ = raw_model_binaryXE()

        model = tf.keras.Model(inputs=input_layer, outputs=[output_layer])
    return model


def model_binaryXE_mid(use_patient_data=False):
    if use_patient_data:
        input_layer, input_semantic, output_layer, image_section_layer_feature = raw_model_binaryXE(use_patient_data)

        model = tf.keras.Model(inputs=[input_layer, input_semantic],
                               outputs=[output_layer, image_section_layer_feature])
    else:
        input_layer, output_layer, image_section_layer_feature = raw_model_binaryXE(use_patient_data)

        model = tf.keras.Model(inputs=input_layer, outputs=[output_layer, image_section_layer_feature])
    return model

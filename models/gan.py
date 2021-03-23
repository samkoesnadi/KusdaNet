"""
Normal model with binary XE as loss function
"""
from common_definitions import *
from utils.weightnorm import WeightNormalization


class EndBlock(tf.keras.layers.Layer):
    def __init__(self, filters, name, **kwargs):
        super().__init__(**kwargs)
        self.sep_conv = tf.keras.layers.SeparableConv2D(filters, kernel_size=3, padding="same",
                                                        kernel_initializer=KERNEL_INITIALIZER, use_bias=False)

        self._bn = tf.keras.layers.BatchNormalization(name=name + "_bn")
        self._act = tf.keras.layers.Activation(GLOBAL_ACTIVATION, name=name + "_act")
        self._name = name
        self._weights = self.sep_conv.weights

    def call(self, inputs, training=False, **kwargs):
        sep_conv = self.sep_conv(inputs)
        _bn = self._bn(sep_conv, training)
        _act = self._act(_bn)

        return _act


class GANModel(tf.keras.Model):
    def __init__(self):
        super(GANModel, self).__init__()

        self.input_layer = tf.keras.layers.Input(shape=(IMAGE_INPUT_SIZE, IMAGE_INPUT_SIZE, 1), name="input_img")

        image_section_model = tf.keras.applications.xception.Xception(include_top=False, weights=None, pooling=None,
                                                                      input_tensor=self.input_layer)
        self._add_layer = image_section_model.layers[125].output

        self.shared_model = tf.keras.Model(inputs=self.input_layer, outputs=self._add_layer)

        self.sep_conv1_act = EndBlock(1536, "block14_sepconv1")
        self.sep_conv2 = tf.keras.layers.SeparableConv2D(2048, kernel_size=3, name="block14_sepconv2",
                                                         padding="same",
                                                         kernel_initializer=KERNEL_INITIALIZER, use_bias=False)

        # post-process the image features
        self._bn = tf.keras.layers.BatchNormalization(
            name="block14_sepconv2_bn")  # the input can be from source or mixed
        self._act = tf.keras.layers.Activation(LAST_ACTIVATION, name="block14_sepconv2_act")

        self.image_section_layer = tf.keras.layers.GlobalAveragePooling2D()

        self.final_do = tf.keras.layers.Dropout(DROPOUT_N)

        self.output_layer = tf.keras.layers.Dense(NUM_CLASSES, activation="sigmoid", kernel_initializer=KERNEL_INITIALIZER)

        if USE_WN:
            self.output_layer = WeightNormalization(self.output_layer, data_init=False)

    @tf.function
    def call_w_features(self, inputs, training=False, **kwargs):
        return self.call_w_everything(inputs, training, **kwargs)[:2]

    def call_w_everything(self, inputs, training=False, **kwargs):
        shared_layer = self.shared_model(inputs, training)

        sep_conv1_act = self.sep_conv1_act(shared_layer, training)
        sep_conv2 = self.sep_conv2(sep_conv1_act)

        # post-process the image features
        _bn = self._bn(sep_conv2, training)  # the input can be from source or mixed
        _act = self._act(_bn)

        image_section_layer = self.image_section_layer(_act)

        final_do = self.final_do(image_section_layer, training)

        output_layer = self.output_layer(final_do)

        return output_layer, image_section_layer, _act

    @tf.function
    def call(self, inputs, training=False, **kwargs):
        return self.call_w_features(inputs, training, **kwargs)[0]


# def model_binaryXE_mid_gan():
#     layers = raw_model_binaryXE_gan()
#
#     model = tf.keras.Model(inputs=layers[0],
#                            outputs=layers[1:])
#     return model
#
#
# def model_binaryXE_gan():
#     layers = raw_model_binaryXE_gan()
#
#     model = tf.keras.Model(inputs=layers[0], outputs=layers[1])
#     return model

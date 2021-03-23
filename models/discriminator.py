from common_definitions import *


def make_discriminator_model():
    input_1 = tf.keras.Input(shape=2048)
    input_2 = tf.keras.Input(shape=NUM_CLASSES)

    hidden_1 = tf.keras.layers.Dense(6144, use_bias=False)(input_1)
    hidden_1_bn = tf.keras.layers.BatchNormalization()(hidden_1)
    hidden_1_act = tf.keras.layers.Activation(GLOBAL_ACTIVATION)(hidden_1_bn)

    hidden_2 = tf.keras.layers.Dense(1024, use_bias=False)(input_2)
    hidden_2_bn = tf.keras.layers.BatchNormalization()(hidden_2)
    hidden_2_act = tf.keras.layers.Activation(GLOBAL_ACTIVATION)(hidden_2_bn)

    hidden = tf.keras.layers.Concatenate()([hidden_1_act, hidden_2_act])

    hidden = tf.keras.layers.Dropout(DROPOUT_N)(hidden)

    output = tf.keras.layers.Dense(NUM_CLASSES, activation="sigmoid")(hidden)

    return tf.keras.Model(inputs=[input_2, input_1], outputs=output)


def make_ADDA_discriminator_model():
    input_1 = tf.keras.Input(shape=2048)

    hidden = tf.keras.layers.Dense(1000, activation=tf.nn.swish)(input_1)
    hidden = tf.keras.layers.Dense(500, activation=tf.nn.swish)(hidden)

    # hidden = tf.keras.layers.Dropout(DROPOUT_N)(hidden_1_act)

    output = tf.keras.layers.Dense(NUM_CLASSES, activation="sigmoid")(hidden)

    return tf.keras.Model(inputs=input_1, outputs=output)


if __name__ == "__main__":
    pass

import tensorflow as tf


def get_output_layers(model, layers):
    output_layers = []
    output_names = []
    for layer in model.layers:
        if layer.name in layers:
            output_layers.append(layer)
            output_names.append(layer.name)

    return output_layers, output_names


def add_weight_decay(model, weight_decay):
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D) or isinstance(layer, tf.keras.layers.Dense):
            layer.add_loss(lambda: tf.keras.regularizers.l2(
                weight_decay)(layer.kernel))
        if hasattr(layer, 'bias_regularizer') and layer.use_bias:
            layer.add_loss(lambda: tf.keras.regularizers.l2(
                weight_decay)(layer.bias))

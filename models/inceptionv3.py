"""Model definition"""

import keras


def inception_model(layer_names):
    """Creates a inception model that returns a list of intermediate output values.
      args:
      layer_names: a list of strings, representing the names of the desired content and style layers

    returns:
      A model that takes the regular inception v3 input and outputs just the content and style layers.

    """

    # Without the fully-connected layer at the top (output) of the network
    inception = keras.applications.InceptionV3(include_top=False, weights="imagenet")

    # We are not training the network
    inception.trainable = False

    output_layers = [inception.get_layer(name).output for name in layer_names]

    model = keras.Model(inputs=inception.input, outputs=output_layers)

    return model

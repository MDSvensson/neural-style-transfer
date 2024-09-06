"""Implements neural style transfer to create a stylized image from content and style images"""

import tensorflow as tf
import keras
from keras import backend as K
from IPython.display import display as display_fn
from IPython.display import clear_output
from models.inceptionv3 import inception_model
from utils.utils import (
    preprocess_image,
    tensor_to_image,
    clip_image_values,
    load_images,
)


def compute_style_loss(features, targets):
    """Calculates style loss between two tensors of shape (height, width, channels)

    Args:
      features: tensor of shape (height, width, channels)
      targets: tensor of shape (height, width, channels)

    Returns:
      Style loss as a scalar value
    """
    style_loss = tf.reduce_mean(tf.square(features - targets))
    return style_loss


def compute_content_loss(features, targets):
    """Calculates content loss between two tensors of shape (height, width, channels)

    Args:
      features: tensor of shape (height, width, channels)
      targets: tensor of shape (height, width, channels)

    Returns:
      Content loss as a scalar value
    """
    content_loss = 0.5 * tf.reduce_sum(tf.square(features - targets))
    return content_loss


def compute_gram_matrix(input_tensor):
    """Computes the Gram matrix of a tensor and normalizes by the number of locations

    Args:
      input_tensor: tensor with dimensions (batch, height, width, channels)

    Returns:
      Normalized Gram matrix
    """
    # this a way to calculate the gram matrices efficiently.
    # more about the einstein notation here: https://ajcr.net/Basic-guide-to-einsum/
    gram = tf.linalg.einsum("bijc,bijd->bcd", input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    height = input_shape[1]
    width = input_shape[2]
    # scale b/c the bigger the filter the more terms in the sum
    # and the bigger the values in the gram matrix but we care
    # only about the correlations between the features
    num_locations = tf.cast(height * width, tf.float32)
    scaled_gram = gram / num_locations
    return scaled_gram


def extract_style_features(image):
    """Extracts style features from an image

    Args:
      image: input image

    Returns:
      List of Gram matrices representing style features
    """
    preprocessed_style_image = preprocess_image(image)
    outputs = inception(preprocessed_style_image)
    style_outputs = outputs[NUM_CONTENT_LAYERS:]
    gram_style_features = [
        compute_gram_matrix(style_layer) for style_layer in style_outputs
    ]
    return gram_style_features


def extract_content_features(image):
    """Extracts content features from an image

    Args:
      image: input image

    Returns:
      Content features extracted from the image
    """
    preprocessed_content_image = preprocess_image(image)
    outputs = inception(preprocessed_content_image)
    content_outputs = outputs[:NUM_CONTENT_LAYERS]
    return content_outputs


def combine_style_content_loss(
    style_targets,
    style_outputs,
    content_targets,
    content_outputs,
    style_weight,
    content_weight,
):
    """Combines style and content loss into a total loss

    Args:
      style_targets: style features of the style image
      style_outputs: style features of the generated image
      content_targets: content features of the content image
      content_outputs: content features of the generated image
      style_weight: weight applied to style loss
      content_weight: weight applied to content loss

    Returns:
      Total combined loss as a scalar value
    """
    style_loss = tf.add_n(
        [
            compute_style_loss(style_output, style_target)
            for style_output, style_target in zip(style_outputs, style_targets)
        ]
    )
    content_loss = tf.add_n(
        [
            compute_content_loss(content_output, content_target)
            for content_output, content_target in zip(content_outputs, content_targets)
        ]
    )
    style_loss *= style_weight / NUM_STYLE_LAYERS
    content_loss *= content_weight / NUM_CONTENT_LAYERS
    total_loss = style_loss + content_loss
    return total_loss


def compute_gradients(
    image, style_targets, content_targets, style_weight, content_weight
):
    """Computes gradients of the total loss with respect to the generated image

    Args:
      image: generated image
      style_targets: style features of the style image
      content_targets: content features of the content image
      style_weight: weight applied to style loss
      content_weight: weight applied to content loss

    Returns:
      Gradients of the total loss with respect to the input image
    """
    with tf.GradientTape() as tape:
        style_features = extract_style_features(image)
        content_features = extract_content_features(image)
        loss = combine_style_content_loss(
            style_targets,
            style_features,
            content_targets,
            content_features,
            style_weight,
            content_weight,
        )
    gradients = tape.gradient(loss, image)
    return gradients


def apply_style_to_image(
    image, style_targets, content_targets, style_weight, content_weight, optimizer
):
    """Applies style to the generated image by updating it based on gradients

    Args:
      image: generated image
      style_targets: style features of the style image
      content_targets: content features of the content image
      style_weight: weight applied to style loss
      content_weight: weight applied to content loss
      optimizer: optimizer for updating the input image
    """
    gradients = compute_gradients(
        image, style_targets, content_targets, style_weight, content_weight
    )
    optimizer.apply_gradients([(gradients, image)])
    image.assign(clip_image_values(image, min_value=0.0, max_value=255.0))


def perform_style_transfer(
    style_image,
    content_image,
    style_weight=1e-2,
    content_weight=1e-4,
    optimizer="adam",
    epochs=1,
    steps_per_epoch=1,
):
    """Executes the neural style transfer process

    Args:
      style_image: image to extract style features from
      content_image: image to apply style to
      style_targets: style features of the style image
      content_targets: content features of the content image
      style_weight: weight applied to style loss
      content_weight: weight applied to content loss
      optimizer: optimizer for updating the input image
      epochs: number of epochs for training
      steps_per_epoch: number of steps to perform per epoch

    Returns:
      Generated image after the final epoch and a collection of generated images per epoch
    """
    images = []
    step = 0
    style_targets = extract_style_features(style_image)
    content_targets = extract_content_features(content_image)
    generated_image = tf.cast(content_image, dtype=tf.float32)
    generated_image = tf.Variable(generated_image)
    images.append(content_image)
    for _ in range(epochs):
        for m in range(steps_per_epoch):
            step += 1
            apply_style_to_image(
                generated_image,
                style_targets,
                content_targets,
                style_weight,
                content_weight,
                optimizer,
            )
            print(".", end="")
            if (m + 1) % 10 == 0:
                images.append(generated_image)
        clear_output(wait=True)
        display_image = tensor_to_image(generated_image)
        display_fn(display_image)
        images.append(generated_image)
        print(f"Train step: {step}")
    generated_image = tf.cast(generated_image, dtype=tf.uint8)
    return generated_image, images


if __name__ == "__main__":
    style_path = keras.utils.get_file(
        "style_image.jpg",
        "file:./data/raw/siemens_accelerator.webp",
    )
    content_path = keras.utils.get_file(
        "content_image.jpg",
        "file:./data/raw/siemens_logo.jpg",
    )
    content_layers = ["conv2d_88"]
    style_layers = ["conv2d", "conv2d_1", "conv2d_2", "conv2d_3", "conv2d_4"]
    content_and_style_layers = content_layers + style_layers
    NUM_CONTENT_LAYERS = len(content_layers)
    NUM_STYLE_LAYERS = len(style_layers)
    K.clear_session()
    inception = inception_model(content_and_style_layers)
    STYLE_WEIGHT = 1
    CONTENT_WEIGHT = 1e-32
    adam = keras.optimizers.Adam(
        keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=80.0, decay_steps=100, decay_rate=0.80
        )
    )
    content_image, style_image = load_images(content_path, style_path)
    stylized_image, display_images = perform_style_transfer(
        style_image=style_image,
        content_image=content_image,
        style_weight=STYLE_WEIGHT,
        content_weight=CONTENT_WEIGHT,
        optimizer=adam,
        epochs=10,
        steps_per_epoch=100,
    )
    STYLIZED_IMAGE_PATH = "./data/results/stylized_image.png"
    stylized_image_png = tf.image.encode_png(stylized_image[0])
    tf.io.write_file(STYLIZED_IMAGE_PATH, stylized_image_png)
    print(f"Stylized image saved to {STYLIZED_IMAGE_PATH}")

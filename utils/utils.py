"""Utilities for image processing"""

import tensorflow as tf
import keras
import matplotlib.pyplot as plt


def tensor_to_image(tensor):
    """converts a tensor to an image"""
    tensor_shape = tf.shape(tensor)
    number_elem_shape = tf.shape(tensor_shape)
    if number_elem_shape > 3:
        assert tensor_shape[0] == 1
        tensor = tensor[0]
    return keras.preprocessing.image.array_to_img(tensor)


def load_img(path_to_img):
    """loads an image as a tensor and scales it to 512 pixels"""
    max_dim = 512
    image = tf.io.read_file(path_to_img)
    image = tf.image.decode_jpeg(image)
    image = tf.image.convert_image_dtype(image, tf.float32)
    shape = tf.shape(image)[:-1]
    shape = tf.cast(tf.shape(image)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim
    new_shape = tf.cast(shape * scale, tf.int32)
    image = tf.image.resize(image, new_shape)
    image = image[tf.newaxis, :]
    image = tf.image.convert_image_dtype(image, tf.uint8)
    return image


def load_images(content_path, style_path):
    """loads the content and path images as tensors"""
    content_image = load_img(f"{content_path}")
    style_image = load_img(f"{style_path}")
    return content_image, style_image


def imshow(image, title=None):
    """displays an image with a corresponding title"""
    if len(image.shape) > 3:
        image = tf.squeeze(image, axis=0)
    plt.imshow(image)
    if title:
        plt.title(title)


def show_images_with_objects(images, titles=None):
    """displays a row of images with corresponding titles"""
    if titles is None:
        titles = []
    if len(images) != len(titles):
        return
    plt.figure(figsize=(20, 12))
    for idx, (image, title) in enumerate(zip(images, titles)):
        plt.subplot(1, len(images), idx + 1)
        plt.xticks([])
        plt.yticks([])
        imshow(image, title)


def clip_image_values(image, min_value=0.0, max_value=255.0):
    """clips the image pixel values by the given min and max"""
    return tf.clip_by_value(image, clip_value_min=min_value, clip_value_max=max_value)


def preprocess_image(image):
    """preprocesses a given image to use with Inception model"""
    image = tf.cast(image, dtype=tf.float32)
    image = (image / 127.5) - 1.0
    return image

import numpy as np


def to_grayscale(img):
    return np.mean(img, axis=2).astype(np.uint8)


def downsample(img, factor=2):
    return img[::factor, ::factor]


def preprocess(img):
    return to_grayscale(downsample(img))

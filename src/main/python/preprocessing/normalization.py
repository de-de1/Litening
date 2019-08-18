from keras.preprocessing.image import img_to_array


def normalize_image(img):
    """Image transformation to numpy array and pixel values normalization to range (0, 1)

    Args:
        img (PIL.Image.Image): Loaded image
    Returns:
        normalized_img (numpy.ndarray): Normalized image
    """

    normalized_img = img_to_array(img) / 255.

    return normalized_img


if __name__ == "__main__":
    pass
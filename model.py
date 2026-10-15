"""Model loading, image preprocessing and prediction helpers.

The Keras model is loaded once when this module is imported so that
every request reuses the same model instead of reloading it.
"""
from keras.models import load_model
from keras.utils import img_to_array
import numpy as np
from PIL import Image

# Loading model
model = load_model("digit_model.h5")


def preprocess_img(img_path):
    """Load an image and convert it into the model's input format.

    Args:
        img_path: A file path or file-like object containing the image.

    Returns:
        A numpy array of shape (1, 224, 224, 3) with values scaled to [0, 1].
    """
    op_img = Image.open(img_path)
    img_resize = op_img.resize((224, 224))
    img2arr = img_to_array(img_resize) / 255.0
    img_reshape = img2arr.reshape(1, 224, 224, 3)
    return img_reshape


def predict_result(predict):
    """Run the model on a preprocessed image and return the predicted digit.

    Args:
        predict: Array returned by preprocess_img().

    Returns:
        The index of the most likely class, which is the predicted digit (0-9).
    """
    pred = model.predict(predict)
    return np.argmax(pred[0], axis=-1)

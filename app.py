"""Flask web application for hand sign digit recognition.

Serves an upload page and shows the digit predicted by the
pre-trained Keras model for the uploaded image.
"""
from flask import Flask, render_template, request

from model import preprocess_img, predict_result

# Instantiating flask app
app = Flask(__name__)


@app.route("/")
def main():
    """Render the home page with the image upload form."""
    return render_template("index.html")


@app.route("/prediction", methods=["POST"])
def predict_image_file():
    """Preprocess the uploaded image and render the predicted digit.

    If no file was sent, or the file is not a readable image, an error
    message is shown on the result page instead of crashing.
    """
    try:
        img = preprocess_img(request.files["file"].stream)
        pred = predict_result(img)
        return render_template("result.html", predictions=str(pred))
    except (KeyError, OSError, ValueError):
        # KeyError: no "file" field in the form
        # OSError: the upload is not a readable image (e.g. a text file)
        # ValueError: the image cannot be reshaped to the model input
        error = "File cannot be processed."
        return render_template("result.html", err=error)


# Driver code
if __name__ == "__main__":
    app.run(port=9000, debug=True)

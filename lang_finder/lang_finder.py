import tensorflow as tf
import tensorflow_text as text

# <--------- Loading Model --------->
# Please note that the model was too large to upload to GitHub,
# I have shared the download link for it in the Readme.md file
# Download and place it in the same file hierarchy as this file
model = tf.keras.models.load_model('language_detector')

# <--------- Languages --------->
# Please note that if there is any change to the list of languages in 
# modelling.py, update this list with the same changes
list_languages = [
    "en",  "ar", "de", "it", "ko", "bg", "da", "el", "fa", "fi", "id", "no", "ro", "sk", "sl", "tl"
    ]

list_languages.sort()

# <--------- Function to run inference on model --------->
# Nothing too complex, it accepts a list of strings,
# returns a list of languages
def find_language(documents):
    documents = tf.constant(documents, tf.string)
    class_probability = model(documents)
    maxval_class = tf.math.argmax(class_probability, axis = -1)

    langs = [list_languages[i] for i in maxval_class.numpy()]

    return langs

import tensorflow as tf
import tensorflow_text as text

model = tf.keras.models.load_model('language_detector')

list_languages = [
    "en",  "ar", "de", "it", "ko", "bg", "da", "el", "fa", "fi", "id", "no", "ro", "sk", "sl", "tl"
    ]

list_languages.sort()

def find_language(documents):
    documents = tf.constant(documents, tf.string)
    class_probability = model(documents)
    maxval_class = tf.math.argmax(class_probability, axis = -1)

    langs = [list_languages[i] for i in maxval_class.numpy()]

    return langs

import tensorflow as tf
import tensorflow_hub as hub

# Use pip install tensorflow-text --no-dependencies, refer link below for why
#https://github.com/tensorflow/text/issues/200
import tensorflow_text as text
import tensorflow_datasets as tfds

SHUFFLE_BUFFER = 10000000
BATCH_SIZE = 16
EPOCHS = 5

list_languages = [
    "en",  "ar", "de", "it", "ko", "bg", "da", "el", "fa", "fi", "id", "no", "ro", "sk", "sl", "tl"
    ]

list_languages.sort()

NUM_LANGS = len(list_languages)

def build_model():
  text_input = tf.keras.layers.Input(shape=(), dtype=tf.string)
  preprocessor = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_multi_cased_preprocess/3")
  encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/4", trainable = False)
  dense1 = tf.keras.layers.Dense(128, activation = 'tanh', initializer='glorot_uniform')
  op = tf.keras.layers.Dense(NUM_LANGS, activation = 'softmax', initializer='glorot_uniform')

  x = preprocessor(text_input)
  x = encoder(x)
  x = x['pooled_output']
  x = dense1(x)
  output = op(x)

  return tf.keras.Model(inputs = text_input, outputs = output, name="language_detector")

model = build_model()

import sklearn
from sklearn.preprocessing import OneHotEncoder

ll = [[i] for i in list_languages]
enc = OneHotEncoder()
enc.fit_transform(ll)

def one_hot(lang):
  return enc.transform([[lang]]).toarray()[0]

def preprocess_lang(text, language):
  text = tf.strings.regex_replace(text, "_START_ARTICLE_ | _START_PARAGRAPH_ | \n | <br> | <p> | </p> | <html> | </html> | <body> | </body>", " ")
  return text, one_hot(language)

list_datasets = []

for language in list_languages:
  list_datasets.append(tfds.load('wiki40b/'+language, split = 'train[0:8000]'))

for i in range(len(list_datasets)):
  list_datasets[i] = list_datasets[i].map(lambda data: preprocess_lang(data['text'], list_languages[i]))

train_db = list_datasets[0]

for i in range(1, len(list_datasets)):
  train_db = train_db.concatenate(list_datasets[i])

train_db = train_db.shuffle(SHUFFLE_BUFFER).batch(BATCH_SIZE)

import time
with tf.device('/GPU:0'):
  loss = tf.keras.losses.BinaryCrossentropy(from_logits = False)
  optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)
  list_step_loss = []
  list_epoch_loss = []
  list_epoch_time = []

with tf.device('/GPU:0'):
  for epoch in range(EPOCHS):
    print(f"<---------------- STARTING EPOCH {epoch} ---------------->")
    strTime = time.time()
    list_step_loss = []

    train_db = train_db.shuffle(SHUFFLE_BUFFER)

    for step, data in enumerate(train_db):
        text = data[0]
        lang = data[1]

        with tf.GradientTape() as tape:
            op = model(text)
            loss_val = loss(lang, op) + tf.constant(1e-8, tf.float32)

        list_step_loss.append(loss_val)

        grads = tape.gradient(loss_val, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        if(step % 200 == 0):
          print(f"Step: {step}, step loss: {loss_val}")

    list_epoch_loss.append(tf.math.reduce_mean(list_step_loss))

    totTime = time.time() - strTime
    list_epoch_time.append(totTime)

    print(f"\nEpoch Loss: {tf.math.reduce_mean(list_step_loss)}")
    print(f"\nTime To Finish Epoch {epoch} - {int(totTime // 60)}:{int(totTime % 60)}\n\n")
    
model.save('language_detector', save_format = 'tf')

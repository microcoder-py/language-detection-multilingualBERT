# Language Detection Using BERT - Base, Cased Multilingual

## Overview
Using the pretrained [BERT Multilingual model](https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/4), a language detection model was devised. The model was fine tuned using the [Wikipedia 40 Billion](https://research.google/pubs/pub49029/) dataset which contains Wikipedia entries from 41 different languages. The model was trained on 16 of the languages. You may find the dataset [here](https://www.tensorflow.org/datasets/catalog/wiki40b).

## Usage

### Prerequisites
* TensorFlow: ```>> pip install tensorflow```
* TensorFlow Hub: ```>> pip install tensorflow-hub```
* TensorFlow Datasets: ```>> pip install tensorflow-datasets```
* TensorFlow Text: ```>> pip install tensorflow-text --no-dependencies ```

> Please note that we are making use of the ```--no-dependencies``` flag because of an error that TensorFlow Text throws pursuant to this following [GitHub Issue](https://github.com/tensorflow/text/issues/200#issuecomment-780998374). If you have already installed TensorFlow text, it is recommended you uninstall and reinstall it

* Sci-kit Learn: ```>> pip install sklearn```

### If you want to perform inference, i.e. simply find what language a given document is written in

* Download the complete repository
* Under the same file hierarchy as the ```lang_finder.py```, download and save the trained model [from this link](https://drive.google.com/drive/folders/1iqByvdbmDkUj-CX8QiVm3IfFLbvuyhvO?usp=sharing)
* Import the file ```lang_finder.py``` and call the function ```lang_finder.find_language([str])``` which accepts a list of strings as input, and returns list of what language they were written in

> **NOTE**: If you changed the set of languages being used, please update the list of languages specified in the file ```lang_finder.py``` as well for it to run correctly

### If you want to train a new model directly within Google Colaboratory:

[Link To Google Colab](https://colab.research.google.com/drive/1kvbc9xU0FLxj6jRn70rzmF6iMn4iOFGY?usp=sharing)

### If you want to train a new model locally

Download the whole repository and run the file ```modelling.py``` with the command
```python3
>> python modelling.py
```

### If you want to train it on more, or different languages

You can find the list of languages available under the Wiki40B dataset in [this link](https://www.tensorflow.org/datasets/catalog/wiki40b). Simply add the languages to the list ```list_languages``` in the file ```modelling.py``` and run it, everything else is configured to work automatically

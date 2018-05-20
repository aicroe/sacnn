# Sentiment Analysis with a Convolutional Neural Network

Based on [Kim Yoon 2014](https://github.com/yoonkim/CNN_sentence) using the embeddings from [Cristian Cardellino 2016](http://crscardellino.me/SBWCE/)

## Requirements

* Tensorflow 1.6 or greater
* Pandas
* Gensim

## Place the raws

Place the raw data inside the raw folder, if doesn't exists create it.

It is looked for a _comments.csv_ file.

It is looked for a embedding bin called _SBW-vectors-300-min5.bin_.

## Preprocess data

Run the script:

```bash
$ python process_data.py
```

It will place the train, validation and test data in the _data_ folder.

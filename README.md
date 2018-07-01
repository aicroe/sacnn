# Sentiment Analysis with a Convolutional Neural Network

Based on [Kim Yoon 2014](https://github.com/yoonkim/CNN_sentence) using the embeddings from [Cristian Cardellino 2016](http://crscardellino.me/SBWCE/)

## Requirements

* Tensorflow 1.6 or greater

Check requeriments.txt for the others

## The workspace

The workspace is in ***~/.scann***, all folders mentioned after are placed there.

## Place the raws

Place the raw data inside the raw folder, if doesn't exists create it.

It is expected a file called _comments.csv_.

It is expected a embedding binary called _SBW-vectors-300-min5.bin_.

## Preprocess data

Run the script:

```bash
$ python process_data.py
```

It will place the train, validation and test data in the _data_ folder.

Optionally it can be used the next script:

```bash
$ python reduce_labels.py
```

In order to decrease the number of labels since the raw data has 5 classes, this script reduce and saves them at _data\_reduced_ folder.

## Train the network

Run the script:

```bash
$ python train.py
```

It reads the data saved at _data_ in order to train the network. Saves the trained parameters a folder called equals as the network.

## Evaluate

Run the script:

```bash
$ python eval.py
```

It reads the trained parameters and evaluates over the test data.
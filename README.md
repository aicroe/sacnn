# Sentiment Analysis with a Convolutional Neural Network

Based on [Kim Yoon 2014](https://github.com/yoonkim/CNN_sentence) using the embeddings from [Cristian Cardellino 2016](http://crscardellino.me/SBWCE/)

## Setup

### Install dependencies
```bash
pip install -r requeriments.txt
```

### Place the raws

Place the data at **~/.sacnn/raw** folder, if doesn't exists create it.
It is expected a file called **comments.csv** and a binary with the embeddings **SBW-vectors-300-min5.bin**.

### Preprocess data

Run the script:

```bash
$ python process_data.py
```

It will place the train, validation and test data at the folder **~/.sacnn/data**.

Optionally it can be used the next script:

```bash
$ python reduce_labels.py
```

In order to decrease the number of labels since the raw data has 5 classes, this script reduce and saves them at **~/.sacnn/data\_reduced**.

## Batch training and evaluation

### Training

The file **hyperparams_list.py** contains the configuration of a list of models to be trained. The script **train.py** will read the hyperparams file and train each one of the models described there.

```bash
$ python train.py
```

This reads the data saved previously at **~/.sacnn/data**. The trained networks will be saved at **~/.sacnn/\<model-name\>**.

### Evaluate

It is posible to run the evaluation of the models after they were trained. The script **eval.py** will read the hyperparams file and restore each model previously trained and run an evaluation over them.

```bash
$ python eval.py
```

## The app

The preprocess data stage must be done before in order this to work.

You can easly up the app running the **server.py** script.

```bash
$ python server.py
```

Or build and run the docker container.
```bash
# Build
$ docker build . -t sacnn:1.0
```

```bash
# Run the container
$ docker run -p 5000:80 --mount type=bind,source=$HOME/.sacnn,target=/root/.sacnn scann
```

For develop you may want to bind the source code to the container too.
```bash
# Bind the source code
$ docker run -p 5000:5000 --mount type=bind,source=$HOME/.sacnn,target=/root/.sacnn --mount type=bind,source="$(pwd)",target=/src sacnn:1.0
```

Once the image is built you could run the app using **docker-compose**.
```bash
# Run using docker-compose
$ docker-compose up
```

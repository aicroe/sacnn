# a Sentiment Analysis Convolutional Neural Network

Based on [Kim Yoon 2014](https://github.com/yoonkim/CNN_sentence) using the embeddings from [Cristian Cardellino 2016](http://crscardellino.me/SBWCE/)

## Setup

**Required python 3.6 or higher.**

* Clone the project
* Create a virtual environment:

```bash
cd sacnn/
python3 -m venv python3
source python3/bin/activate
```

* Install for development
```bash
python setup.py develop
```

* Install for production?
```bash
python setup.py install
```

This will install the project in a way that would let execute the project's command line scripts as a regular console program. The last script though would need special considerations, there is a different recommended way to execute it (TODO: add these steps).

**Place the raws.**

Place the data at the folder **~/.sacnn/raw**, if doesn't exists create it.
It is expected a file called **comments.csv** which would contain the whole labeled data set, and a binary that holds the embeddings **SBW-vectors-300-min5.bin**.

## Command Line Scripts

When the project is installed, some scripts will become available in the system command line:

**Process the data set**

Process and store the samples from **comments.csv** in a format easier to use for Machine Learning algorithms.

```bash
$ sacnn_process_data
```

It will place the train, validation and test data sets at the folder **~/.sacnn/data**.

Optionally, after the previous script run, the next can be used to re-process the data in order to have sets with a slightly different configuration (it can be used to train a wider range of models):

```bash
$ sacnn_reduce_labels
```

Reduces the data labels number from 5 to 3. The data will be saved at  **~/.sacnn/data\_reduced**.

**Training**

The train script, same as the previous two, would be available in the command line after installation. This script reads a number of model configurations (hyper-parameters) in a json format from the standard input in order to build, train, measure and save said models.

It is recommended to have this configuration in a `.json` file and pass it to the train script through the redirection operator `<`.

The way the needed configuration data should be arranged in a json file is the next:

```json
[
  {
    "name": "<string>", // instance's name identifier
    "arch": "<kim|kim1fc|kimpc|kim1fcpc>",
    "trainer": "<simple|sgd|early_stop|sgd_early_stop>",
    "learning_rate": "<float>",
    "epochs": "<int>",
    "validation_gap": "<int>",
    "minibatch_size": "<int>",
    "keep_prob": "<float>", // a number between 0 and 1
    "filters": "<[[filter_height, num_filters]]>",
    "hidden_units": "<int>", // Only kim1fc and kim1fcpc need it
    "num_labels": "<int>" // 3 or 5
  }
]
```

Suppose the configuration is saved in file called `hyperparams_list.json`, the next sentence should do the work.

```bash
$ sacnn_train < hyperparams_list.json
```

It will read the previously saved data at **~/.sacnn/\<data|data_reduced\>** in order to train the models according to each configuration. The trained models will be saved at **~/.sacnn/\<arch-name\>**.

If you wish to save the logs, you may want to try this sentence instead.

```bash
$ sacnn_train < hyperparams_list.json > train-$(date +%Y%m%d%H%M).log
```

**Evaluation**

It is possible to run a number of additional measures after after the models were trained. This script will read the models configuration, the same way as the previous training script, and restore each one of them to run the evaluation (again supposing the configuration is saved in a file called `hyperparams_list.json`).

```bash
$ sacnn_eval < hyperparams_list.json
```

## Service and Web UI

This project is bundled with a web application that allows to train and test the models this work exposes through a simple web interface backed by a custom service.

TODO: The next is un-accurate information, it needs to be updated

The preprocess data stage must be done before in order this to work.

You can easily up the app running the **server.py** script.

```bash
$ python server.py
```

Or build and run the image in an docker container.
```bash
# Build the image
$ docker build . -t sacnn:1.0
```

```bash
# Run the image on a container
$ docker run -p 5000:80 --mount type=bind,source=$HOME/.sacnn,target=/root/.sacnn scann
```

For develop you may want to bind the source code to the container too.
```bash
# Bind the source code
$ docker run -p 5000:5000 --mount type=bind,source=$HOME/.sacnn,target=/root/.sacnn --mount type=bind,source="$(pwd)",target=/src sacnn:1.0
```

Once the image is built you could just run the app using **docker-compose**.
```bash
# Run using docker-compose
$ docker-compose up
```

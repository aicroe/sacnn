# a Sentiment Analysis Convolutional Neural Network

Based on [Kim Yoon 2014](https://github.com/yoonkim/CNN_sentence) using the word embeddings from [Cristian Cardellino 2019](https://crscardellino.github.io/SBWCE/)

## Setup

Required python 3.6 or higher. Unix based environment recommended.

### Steps

* Clone the repository
* Place the raws

The data (word embedding and comments data set) must be placed at the folder **~/.sacnn/raw**, if it doesn't exists create it.

It is expected a file called **comments.csv** which would contain the whole labeled data set, and a binary that holds the embeddings **SBW-vectors-300-min5.bin**. Look at the **raws/** folder on this repository in order to get those files.

**For production**

* Install the dependencies
```bash
$ pip install -r requirements.txt
```

* Run the install script
```bash
$ python setup.py install
```

This installs the project (along with its dependencies) and exposes a couple of scripts that would be available from the command line interface.

**For development**

* Create a dedicated virtual environment:

```bash
$ cd sacnn/
$ python3 -m venv python3
$ source python3/bin/activate
```

* Install for development
```bash
$ python setup.py develop
```

**Troubleshooting**

Due the complexity of some libraries this project relies on, the automatic dependency installation through `setup.py` script is weak. That's why it is recommended to set up the dependencies via `pip install` first, but even that could fail on different python versions. If you encounter problems during installation, you may need to manually install the libraries in conflict.

## Command Line Scripts

When the project is installed, some scripts will become available from the command line interface:

**Process the data set**

Reads and processes the samples from **comments.csv**, then saves it in a format propitious to feed a Machine Learning algorithm.

```bash
$ sacnn_process_data
```

It will place the *train*, *validation* and *test* data sets at **~/.sacnn/data**.

Optionally, after the previous script run, the next can be used to re-process the data in order to build new data sets with a slightly different configuration (this can be used to train a wider range of models):

```bash
$ sacnn_reduce_labels
```

Reduces the data labels range from 5 to 3. The new sets will be saved at  **~/.sacnn/data\_reduced**.

**Training**

The train script, same as the previous two, would be available from the command line after installation. This script reads several model configurations (hyper-parameters) in a json format from the standard input in order to *build*, *train*, *measure* and *save* said models.

It is recommended to have this configuration in a `.json` file and pass it to the train script with the redirection operator `<`.

The way the configuration file should be arranged is the next:

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

Suppose the configuration is saved in file called `hyperparams_list.json`, the next sentence should do the job.

```bash
$ sacnn_train < hyperparams_list.json
```

It will read the previously saved data at **~/.sacnn/\<data|data_reduced\>** in order to train the models, according to each configuration. The trained models will be saved at **~/.sacnn/\<arch-name\>**.

If you wish to save the logs, you may want to run this sentence instead.

```bash
$ sacnn_train < hyperparams_list.json > train-$(date +%Y%m%d%H%M).log
```

**Evaluation**

It is possible to run additional measures after the models were trained. This script can read the models configuration, the same way as the previous script. It restores each trained model to run an evaluation (the next supposes the configuration is saved in a file called `hyperparams_list.json`).

```bash
$ sacnn_eval < hyperparams_list.json
```

## Web App

The project is bundled with a web application that allows train and test the models this work exposes, through a web client supported by a custom server.

Note: Data preprocessing must be done beforehand in order this to work.

Once the project is set up, you can easily start the app by running the server's script.

```bash
$ sacnn_server
```

Or build a the docker image and then run it.
```bash
# Build the image
$ cd sacnn/
$ docker build . -t sacnn:1.0
```

```bash
# Run the image on a container
$ docker run -p 5000:5000 --mount type=bind,source=$HOME/.sacnn,target=/root/.sacnn sacnn:1.0
```

Once the image is built you could just run the app using **docker-compose**.
```bash
# Run using docker-compose
$ docker-compose up
```

import threading
import traceback
import logging

from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.exceptions import BadRequest

from sacnn.app import QueryController, ClassifierController, TrainController

logging.basicConfig(
    format='%(asctime)s %(message)s',
    datefmt='%d/%m/%Y %I:%M:%S %p',
    level=logging.INFO,
)
app = Flask(__name__)
query_ctrl = QueryController()
classifier_ctrl = ClassifierController()
train_ctrl = TrainController()

@app.route('/')
@app.route('/classify', methods=['GET'])
def render_classify():
    names = query_ctrl.get_instances_names()
    return render_template('classifier.html', instance_names=names)


@app.route('/classify', methods=['POST'])
def classify_comment():
    content = request.get_json()
    comments = content['comments']
    instance_name = content['instance_name']

    if len(comments) <= 0:
        raise BadRequest('empty_comments')

    try:
        results = classifier_ctrl.classify(instance_name, comments)
    except:
        traceback.print_exc()
        raise BadRequest('classifying_error')

    return jsonify(results=results)


@app.route('/train')
def render_train():
    return render_template('train.html')


@app.route('/train/<instance_name>/learning_curve', methods=['GET'])
def return_learning_curve_img(instance_name):
    try:
        instance_state = train_ctrl.get_training_instance(instance_name)
    except:
        traceback.print_exc()
        raise BadRequest('instance_name_not_found')

    filepath = instance_state.learning_curve_path
    return send_file(filepath, mimetype='image/png')


@app.route('/train', methods=['POST'])
def train_instance():
    hyperparams = request.get_json()

    def train_instance_runner():
        train_ctrl.train_instance(hyperparams)

    name = hyperparams['name']
    if not query_ctrl.is_unique_instance_name(name):
        raise BadRequest('not_unique_name')

    threading.Thread(target=train_instance_runner).start()
    return jsonify(success="success")


@app.route('/train_state', methods=['POST'])
def train_instance_state():
    params = request.get_json()

    try:
        state = train_ctrl.get_training_instance(params['name']).to_dict()
    except:
        traceback.print_exc()
        raise BadRequest('instance_not_found')

    return jsonify(state)


def main():
    app.run(host='0.0.0.0')

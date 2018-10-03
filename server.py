from app.app_state import AppState
from app.classifier_controller import ClassifierController
from app.train_controller import TrainController
from flask import Flask
from flask import render_template
from flask import request
from flask import jsonify
from werkzeug.exceptions import BadRequest
import threading


app = Flask(__name__)
app_state = AppState()
classifier_ctrl = ClassifierController(app_state)
train_ctrl = TrainController(app_state)


@app.route('/')
@app.route('/classify', methods=['GET'])
def render_classify():
    return render_template('classifier.html', instance_names=app_state.get_instance_names())


@app.route('/train')
def render_train():
    return render_template('train.html')


@app.route('/classify', methods=['POST'])
def classify_comment():
    content = request.get_json()
    comment = content['comment']
    instance_name = content['instance_name']
    result = 'Empty comment'
    if len(comment) > 0:
        result = classifier_ctrl.classify(instance_name, comment)
    return jsonify(result=result)


@app.route('/train', methods=['POST'])
def train_instance():
    hyperparams = request.get_json()

    def train_instance_runner():
        train_ctrl.train_instance(hyperparams)

    if not app_state.is_unique_name(hyperparams['name']):
        raise BadRequest('not_unique_name')
    threading.Thread(target=train_instance_runner).start()
    print('Training')
    return jsonify(success="success")


@app.route('/train_state', methods=['POST'])
def train_instance_state():
    params = request.get_json()
    state = train_ctrl.get_training_state(params['name']).to_dict()
    print(state)
    return jsonify(state)


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)

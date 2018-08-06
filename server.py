from app.app_state import AppState
from app.classifier_controller import ClassifierController
from flask import Flask
from flask import render_template
from flask import url_for
from flask import request
from flask import jsonify


app = Flask(__name__)
app_state = AppState()
classifier_ctrl = ClassifierController(app_state)


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
    pass


if __name__ == '__main__':
    app.run(debug=True)

FROM tensorflow/tensorflow:1.15.0-py3

WORKDIR /sacnn
COPY . /sacnn

# Workaround to https://github.com/boto/botocore/issues/1872
RUN pip install "python-dateutil<2.8.1"

RUN pip install -r requirements.txt
RUN python setup.py install

EXPOSE 5000

ENV FLASK_ENV production

CMD ["sacnn_server"]
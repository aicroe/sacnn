FROM tensorflow/tensorflow:latest-py3

WORKDIR /src
COPY . /src
RUN pip install -r requeriments.txt

EXPOSE 80

CMD ["python", "server.py"]
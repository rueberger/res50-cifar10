FROM tensorflow/tensorflow:1.10.1-py3

WORKDIR /work

ADD . /work/reslab

RUN pip install -e /work/reslab

ENTRYPOINT jupyter lab --no-browser --port=8888 --ip=0.0.0.0

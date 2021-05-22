FROM pytorch/pytorch:1.8.0-cuda11.1-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -qq && \
    apt-get install -y git vim libgtk2.0-dev && \
    rm -rf /var/cache/apk/*

RUN pip --no-cache-dir install Cython

COPY requirements.txt /workspace/requirements.txt
RUN pip --no-cache-dir install -r /workspace/requirements.txt

COPY download_pretrain.py /workspace/download_pretrain.py
RUN python download_pretrain.py

COPY . /workspace

RUN chmod +x /workspace/launch.sh

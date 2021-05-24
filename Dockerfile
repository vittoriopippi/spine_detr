FROM pytorch/pytorch:1.8.0-cuda11.1-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -qq && \
    apt-get install -y git vim libgtk2.0-dev && \
    rm -rf /var/cache/apk/*

ARG USER_ID=1003
ARG GROUP_ID=1003
RUN echo "USER_ID=${USER_ID} GROUP_ID=${GROUP_ID}"
RUN addgroup --gid ${GROUP_ID} user && adduser --disabled-password --gecos '' --uid ${USER_ID} --gid ${GROUP_ID} user && usermod -a -G root user
USER user
WORKDIR /home/user

RUN pip --no-cache-dir install Cython

COPY requirements.txt ./workspace/requirements.txt
RUN pip --no-cache-dir install -r ./workspace/requirements.txt

COPY download_pretrain.py ./workspace/download_pretrain.py
RUN python download_pretrain.py

COPY . ./workspace

RUN chmod +x ./workspace/launch.sh

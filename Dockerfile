FROM pytorch/pytorch:1.8.0-cuda11.1-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -qq && \
    apt-get install -y git vim libgtk2.0-dev && \
    rm -rf /var/cache/apk/*

# Create a user with defined USER_ID and GROUP_ID. Usually, these variables are the same as the user who builds the docker file, in that way it is possible to modify the data generated inside the container outside of it.
ARG USER_ID
ARG GROUP_ID
RUN echo "USER_ID=${USER_ID} GROUP_ID=${GROUP_ID}"
RUN addgroup --gid ${GROUP_ID} user && adduser --disabled-password --gecos '' --uid ${USER_ID} --gid ${GROUP_ID} user && usermod -a -G root user
WORKDIR /home/user

RUN pip --no-cache-dir install Cython

COPY requirements.txt ./workspace/requirements.txt
RUN pip --no-cache-dir install -r ./workspace/requirements.txt

COPY download_pretrain.py ./workspace/download_pretrain.py

COPY . ./workspace
WORKDIR /home/user/workspace

RUN chmod +x launch.sh
RUN chown -R user /home/user

USER user
RUN python download_pretrain.py
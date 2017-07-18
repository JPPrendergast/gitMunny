FROM ubuntu:16.04

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && apt-get install -y nvidia-cuda-toolkit \
	python3 \
	python-setuptools \
	python3-pip \
	python-tk

COPY requirements.txt /requirements.txt
COPY ta-lib-0.4.0-src.tar.gz /ta-lib.tar.gz 

RUN tar -xvzf ta-lib.tar.gz && \
	chown -hR root /ta-lib/ && \
	cd ta-lib && \
	./configure --prefix=/usr && \
	make -s && \
	make install && \
	cd .. && \
	pip install -r requirements.txt && \
	pip install ta-lib

ADD . /ta-lib/
ADD . /src/
ADD . /data/
RUN cd src && \
	export CUDA_HOME=/opt/cuda && \
	export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$CUDA_HOME/lib64:$CUDA_HOME/extras/CUPTI/lib64"

WORKDIR .

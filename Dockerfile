FROM ubuntu:16.04

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && apt-get install -y python3 \
	python3-setuptools \
	python3-pip \
	python3-tk 

COPY requirements.txt /requirements.txt
COPY ta-lib-0.4.0-src.tar.gz /ta-lib.tar.gz 

RUN tar -xvzf ta-lib.tar.gz && \
	chown -hR root /ta-lib/ && \
	cd ta-lib && \
	./configure --prefix=/usr && \
	make -s && \
	make install && \
	cd .. && \
	pip3 install -r requirements.txt && \
	pip3 install ta-lib

ADD . /gitMunny/
	
WORKDIR /gitMunny/

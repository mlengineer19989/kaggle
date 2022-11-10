
#FROM --platform=linux/amd64 ubuntu:latest
#FROM jupyter/scipy-notebook:ubuntu-20.04

FROM sonoisa/deep-learning-coding:pytorch1.12.0_tensorflow2.9.1 AS deep-learning-base

WORKDIR /usr/src/app
COPY . .

RUN pip install --upgrade pip
RUN pip install -r requirements_docker_m1.txt
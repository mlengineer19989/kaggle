
#FROM --platform=linux/amd64 ubuntu:latest
#FROM jupyter/scipy-notebook:ubuntu-20.04

FROM sonoisa/deep-learning-coding:pytorch1.12.0_tensorflow2.9.1 AS deep-learning-base

RUN pip install --upgrade pip
RUN pip install jupyter

WORKDIR /usr/src/app
COPY . .
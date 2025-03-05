# Base pytorch image with GPU suport
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

# Basic package update
RUN apt-get update && apt upgrade -y

# The flag is set to avoid unnecesarry questions during installation
# Example: It asks to set timezone during instalation of python.
ARG DEBIAN_FRONTEND=noninteractive

# Install VIM (For Convenience)
RUN apt-get install -y vim
RUN apt-get -y install git

# Huggingface installation
RUN pip install transformers
COPY requirements.txt .
RUN pip install -r requirements.txt
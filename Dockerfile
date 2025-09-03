FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04


# set environment variables


# copy project
COPY ./requirements.txt ./

RUN apt-get update -y
RUN apt-get update && apt-get install -y \
    wget \
    build-essential \
    curl \
    software-properties-common \
    ca-certificates \
    libncurses5-dev \
    libgdbm-dev \
    libnss3-dev \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    libffi-dev \
    liblzma-dev \
    tk-dev \
    libgdbm-compat-dev \
    && apt-get clean
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 libpoppler-dev poppler-utils -y && apt-get clean
RUN curl -O https://www.python.org/ftp/python/3.13.0/Python-3.13.0.tgz && \
    tar -xvf Python-3.13.0.tgz && \
    cd Python-3.13.0 && \
    ./configure --enable-optimizations && \
    make && \
    make install && \
    cd .. && \
    rm -rf Python-3.13.0 Python-3.13.0.tgz

# Verify Python installation
RUN python3 --version

# Install pip and any other Python dependencies
RUN curl https://bootstrap.pypa.io/get-pip.py | python3

# Install common Python packages
RUN pip install --upgrade pip
# We need wget to set up the PPA and xvfb to have a virtual screen and unzip to install the Chromedriver

RUN pip install --upgrade pip
RUN pip install -r requirements.txt
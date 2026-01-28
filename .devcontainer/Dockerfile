FROM ubuntu:22.04

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY ../requirements.txt .
RUN pip3 install -r requirements.txt

EXPOSE 8000 9090

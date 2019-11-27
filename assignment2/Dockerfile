FROM nvcr.io/nvidia/tensorflow:19.10-py3
WORKDIR /assignment2
RUN apt-get update && apt-get install -y \
	tmux \
	git \
	cmake \
	zlib1g-dev \
	ffmpeg

RUN pip install gym[atari] \
	gym==0.10.9 \
	matplotlib==2.0.2 \
	numpy==1.14.2 \
	six==1.11.0


FROM althack/ros2:foxy-dev

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

RUN pip3 install --upgrade pip

ADD requirements.txt /
RUN pip3 install -r requirements.txt
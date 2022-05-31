FROM python:3.9.6
WORKDIR /capstone
ADD . /capstone
COPY requirements.txt /capstone
RUN pip install -r requirements.txt
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
CMD ["python","test.py"]

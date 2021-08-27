# set base image (host OS)
FROM python:3.9-slim-buster

# set the working directory in the container
WORKDIR /code

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

# copy the content of the local src directory to the working directory
COPY . .

# command to run on container start
CMD [ "python", "./test.py" ]
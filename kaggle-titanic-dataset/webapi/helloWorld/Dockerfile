FROM python:3.9-slim

# Allow statements and log messages to immediately appear in the Knative logs
ENV PYTHONUNBUFFERED True

# gcc and make are needed for installing fastapi[standard]
RUN apt-get update \
&& apt-get install gcc -y \
&& apt-get install make \
&& apt-get clean

# Copy local code to the container image.
WORKDIR /app
COPY requirements.txt /tmp/

# Install production dependencies.
RUN pip install -r /tmp/requirements.txt
COPY app/ /app/

# Run the web service on container startup. Here we use the gunicorn
# webserver, with one worker process and 8 threads.
# For environments with multiple CPU cores, increase the number of workers
# to be equal to the cores available.
CMD exec gunicorn --bind :$PORT --workers 1 --worker-class uvicorn.workers.UvicornWorker --threads 8 helloWorld:app
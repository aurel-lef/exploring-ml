FROM python:3.8-slim

# Allow statements and log messages to immediately appear in the Knative logs
ENV PYTHONUNBUFFERED True

# gcc and make are needed for installing fastapi[standard]
RUN apt-get update \
&& apt-get install gcc -y \
&& apt-get install make \
&& apt-get clean

COPY requirements.txt /tmp/

RUN pip install -r /tmp/requirements.txt
COPY app /app

# Run the web service on container startup
CMD exec gunicorn --bind :$PORT --workers 1 --worker-class uvicorn.workers.UvicornWorker --threads 8 app.main:app
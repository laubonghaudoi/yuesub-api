FROM python:3.8-slim-buster

RUN apt-get update \
  && apt-get install -y --no-install-recommends \
  gcc build-essential \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

COPY . /app

CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0", "--port=8081"]
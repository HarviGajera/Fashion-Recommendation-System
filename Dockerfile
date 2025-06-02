FROM python:3.10-slim
# FROM pytorch/pytorch:2.7.0-cpu

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY . /app/

RUN apt-get update && apt-get install -y gcc build-essential && \
    pip install --upgrade pip && pip install --cache-dir=/tmp/pip-cache -r requirements.txt

CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
FROM python:3.9.16-bullseye

COPY ./requirements.txt requirements.txt

ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

RUN apt-get update -y && apt-get install -y gdal-bin libgdal-dev python3-gdal && \
    python3 -m pip install --upgrade pip && \
    python3 -m pip install -r requirements.txt

WORKDIR /app

COPY ./src/all-MiniLM-L6-v2 ./all-MiniLM-L6-v2
COPY ./src/bart-large-cnn ./bart-large-cnn
# Add code after model for faster builds
COPY ./src/logging.conf ./logging.conf
COPY ./src/main.py ./main.py
COPY ./src/models.py ./models.py
COPY ./src/model_interfaces.py ./model_interfaces.py

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
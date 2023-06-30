FROM python:3.9.16-bullseye

COPY ./requirements.txt requirements.txt

ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

RUN apt-get update -y && apt-get install -y gdal-bin libgdal-dev python3-gdal && \
    python3 -m pip install --upgrade pip && \
    python3 -m pip install -r requirements.txt

RUN python -c "from transformers import pipeline; print(pipeline('sentiment-analysis')('we love you'))"

# WORKDIR /app

# COPY ./src ./

# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
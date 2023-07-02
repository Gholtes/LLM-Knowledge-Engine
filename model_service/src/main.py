import os
import traceback
import base64
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
import logging
import numpy as np

from models import *

# setup logging
logging.config.fileConfig('logging.conf', disable_existing_loggers=False)
# get logger for this file
logger = logging.getLogger(__name__)

# Start app
app = FastAPI()
logger.info("Fast API app initialised")

# Import and start model

from embeddings_model import EmbeddingsModel
model = EmbeddingsModel()
logger.info("EmbeddingsModel initialised")

APP_HOST = '0.0.0.0'
APP_PORT = 7050

#### MAIN ROUTES ####

@app.post("/embeddings/get", response_model=EmbeddingsGetResponse)
async def embeddings_get(request: EmbeddingsGetRequest):
    """
    Returns the embeddings vector for the text input

    test with curl 
    curl -X POST localhost:7050/embeddings/get -H 'Content-Type: application/json' -d '{"encode":true, "text":"This is a test sentance"}'
    """
    # Response must match spec of the class exampleResponse from ./models.py
    embedding = model.get(request.text)
    if request.encode:
        embeddings_str = encode_nparray(embedding)
    else:
        embeddings_str = embedding.tolist()
    resp = {
        'embeddings': embeddings_str
    }
    return resp

@app.post("/embeddings/get-batch", response_model=EmbeddingsGetBatchResponse)
async def embeddings_get_batch(request: EmbeddingsGetBatchRequest):
    """
    Returns the embeddings vector for the text input

    test with curl 
    curl -X POST localhost:7050/embeddings/get-batch -H 'Content-Type: application/json' -d '{"encode":false, "texts":["This is a test sentance","this is also a test sentance"]}'
    """
    # Response must match spec of the class exampleResponse from ./models.py
    embeddings = model.get_batch(request.texts)
    resp = {
        'embeddings': []
    }
    if request.encode:
        for embedding in embeddings:
            embeddings_str = encode_nparray(embedding)
            resp['embeddings'].append(embeddings_str)
    else:
        for embedding in embeddings:
            resp['embeddings'].append(embedding.tolist())
    logger.info(resp)
    return resp


#### SYSTEM ROUTES ####

@app.get("/")
async def root():
    return RedirectResponse(f'http://{APP_HOST}:{APP_PORT}/docs', status_code=303)


@app.get("/live")
async def live():
    return "LIVE"


@app.get("/ready")
async def ready():
    if not check_ready():
        raise HTTPException(status_code=503, detail="Service not ready")
    return "READY"


@app.post("/examples/typed-reponse-demo", response_model=ExampleResponse)
async def example(request: ExampleRequest):
    """Example of a typed post endpoint. See ./models.py for the definitions of the input and output types

    test with: 
        curl -d '{"age":23, "name":"myUser"}' -H "Content-Type: application/json" -X POST http://localhost:7003/examples/typed-reponse-demo

    Args:
        ExampleRequest (exampleRequest): a users age and name

    Returns:
        ExampleResponse (exampleResponse): a users age and name and isvalid flag
    """
    # Response must match spec of the class exampleResponse from ./models.py
    resp = {
        'age': request.age,
        'name': request.name,
        'id': 'U123456789'
    }
    return resp


def check_ready(): 
    return True

def encode_nparray(arr):
    return base64.b64encode(arr.astype(np.float16).tobytes()).decode('ascii')

if __name__ == "__main__":
    uvicorn.run(app, host=APP_HOST, port=APP_PORT)

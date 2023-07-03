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

from model_interfaces import EmbeddingsModel
embeddings_model= EmbeddingsModel()
logger.info("EmbeddingsModel initialised")

from model_interfaces import SummarisationModel
summarisation_model = SummarisationModel()
logger.info("SummarisationModel initialised")

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
    embedding = embeddings_model.get(request.text)
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
    embeddings = embeddings_model.get_batch(request.texts)
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

@app.post("/summarisation/get", response_model=SummarisationResponse)
async def summarisation_get(request: SummarisationRequest):
    """
    Returns the embeddings vector for the text input

    test with curl 
    curl -X POST localhost:7050/summarisation/get -H 'Content-Type: application/json' -d '{"context":"Pastas are divided into two broad categories: dried (pasta secca) and fresh (pasta fresca). Most dried pasta is produced commercially via an extrusion process, although it can be produced at home. Fresh pasta is traditionally produced by hand, sometimes with the aid of simple machines. Fresh pastas available in grocery stores are produced commercially by large-scale machines. Both dried and fresh pastas come in a number of shapes and varieties, with 310 specific forms known by over 1,300 documented names. In Italy, the names of specific pasta shapes or types often vary by locale. For example, the pasta form cavatelli is known by 28 different names depending upon the town and region. Common forms of pasta include long and short shapes, tubes, flat shapes or sheets, miniature shapes for soup, those meant to be filled or stuffed, and specialty or decorative shapes.As a category in Italian cuisine, both fresh and dried pastas are classically used in one of three kinds of prepared dishes: as pasta asciutta (or pastasciutta), cooked pasta is plated and served with a complementary sauce or condiment; a second classification of pasta dishes is pasta in brodo, in which the pasta is part of a soup-type dish. A third category is pasta al forno, in which the pasta is incorporated into a dish that is subsequently baked in the oven. Pasta dishes are generally simple, but individual dishes vary in preparation. Some pasta dishes are served as a small first course or for light lunches, such as pasta salads. Other dishes may be portioned larger and used for dinner. Pasta sauces similarly may vary in taste, color and texture.", "max_length":250, "min_length":100}'
    """
    # Response must match spec of the class exampleResponse from ./models.py
    summary = summarisation_model.get(request.context, max_length=request.max_length, min_length=request.min_length)
    resp = {
        'summary': summary
    }
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

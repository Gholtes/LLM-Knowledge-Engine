import os
import traceback
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
import logging

logger = logging.getLogger("app")

from models import *
from embeddings_model import EmbeddingsModel

# Start app
app = FastAPI()
logger.info("Fast API app initialised")

model = EmbeddingsModel()

APP_HOST = '0.0.0.0'
APP_PORT = 7000

#### MAIN ROUTES ####

@app.post("/embeddings/get", response_model=EmbeddingsGetResponse)
async def embeddings_get(request: EmbeddingsGetRequest):
    """
    Returns the embeddings vector for the text input

    test with curl localhost:7005/embeddings/get
    """
    # Response must match spec of the class exampleResponse from ./models.py
    logger.info(request.text)
    embeddings = model.get(request.text)
    logger.info(embeddings)
    resp = {
        'embeddings': embeddings
    }
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

if __name__ == "__main__":
    uvicorn.run(app, host=APP_HOST, port=APP_PORT)

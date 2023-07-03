import os
import traceback
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
import logging

from models import *
from interface import Interface

# setup logging
logging.config.fileConfig('logging.conf', disable_existing_loggers=False)
# get logger for this file
logger = logging.getLogger(__name__)

# Start app
app = FastAPI()
logger.info("Fast API app initialised")

interface = Interface()

APP_HOST = '0.0.0.0'
APP_PORT = 7050

#### MAIN ROUTES ####

@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """
    Returns the embeddings vector for the text input

    test with curl 
    curl -X POST localhost:7051/search -H 'Content-Type: application/json' -d '{"query":"What is some useful information abour rainforests?"}'
    """
    # Response must match spec of the class exampleResponse from ./models.py
    resp = {}
    summary, top_matches = interface.search_and_summerise(request.query)
    resp["summary"] = summary
    resp["document_ids"] = top_matches
    return resp

@app.post("/enrol", response_model=EnrolResponse)
async def enrol(request: EnrolRequest):
    """
    Returns the embeddings vector for the text input

    test with curl 
    curl -X POST localhost:7051/enrol -H 'Content-Type: application/json' -d '{"text":"Rainforests are also home to an estimated  50 million people - with more than a billion people depending on them for their livelihoods. Expert knowledge of plants and their properties, of the seasons, and of animal interactions, means local people are the best custodians of tropical forest.  In fact, forest that is managed by Indigenous peoples and local communities has lower deforestation rates than forest that is managed by governments. Recognition of Indigenous land rights and formalization of land ownership is one of the best ways to secure the future of the world’s rainforests.", "source":"www.globalwitness.org"}'
    """
    # Response must match spec of the class exampleResponse from ./models.py
    resp = {}
    status = interface.enrol_document(request.text, request.source)
    resp["status"] = "SUCCESS"
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


def check_ready(): 
    return True

if __name__ == "__main__":
    uvicorn.run(app, host=APP_HOST, port=APP_PORT)

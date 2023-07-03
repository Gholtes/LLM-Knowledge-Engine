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
    curl -X POST localhost:7051/search -H 'Content-Type: application/json' -d '{"query":"This is a test sentance as well"}'
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
    curl -X POST localhost:7051/enrol -H 'Content-Type: application/json' -d '{"text":"Deserts cover more than one-fifth of the Earth's land area, and they are found on every continent. A place that receives less than 10 inches (25 centimeters) of rain per year is considered a desert. Deserts are part of a wider class of regions called drylands. These areas exist under a moisture deficit, which means they can frequently lose more moisture through evaporation than they recieve as rain", "source":"www.globalwitness.org"}'
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

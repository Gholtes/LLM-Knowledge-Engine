import os
import base64
import numpy as np
import json
import requests
import logging
import time

from db import MilvusInterface

logger = logging.getLogger(__name__)

class Interface:
    def __init__(self):
        self.model_service_url = os.getenv("MODEL_SERVICE_URL")
        self.vector_db = MilvusInterface()

    def search_and_summerise(self, query):
        # Get embeddings
        embeddings = self.get_embeddings(query)
        # Get K-nearest documents from vector db
        results = self.vector_db.query(embeddings)
        # Summerise results
        concat_results = "; ".join([r["text_preview"] for r in results])
        logger.info(concat_results)
        summary = self.get_summary(concat_results)
        logger.info(summary)
        return summary, results
    
    def get_embeddings(self, text):
        return self._get_embeddings(text)
    
    def get_summary(self, text):
        return self._get_summary(text)

    def enrol_document(self, text, source_path):
        st = time.time()
        embeddings = self.get_embeddings(text)
        embeddings = embeddings.tolist()
        logger.info("time to get embeddings: "+str(time.time()-st))
        st = time.time()
        self.vector_db.insert([source_path], [text], [embeddings])
        logger.info("time to insert into db: "+str(time.time()-st))
    
    def enrol_documents(self, texts, source_paths):
        """A bulk insert function would be very useful here!"""
        pass
    
    def _get_embeddings(self, text):
        endpoint = "/embeddings/get"
        url = self.model_service_url + endpoint
        payload = {"encode":True, "text":text}
        resp = self._make_post_request(url, payload)
        embeddings = self._decode_b64_numpy(resp["embeddings_string"])
        return embeddings
    
    def _get_summary(self, text):
        endpoint = "/summarisation/get"
        url = self.model_service_url + endpoint
        payload = {"context":text, "max_length":250, "min_length":85}
        resp = self._make_post_request(url, payload)
        summary = resp["summary"]
        return summary
    
    def _make_post_request(self, url, payload):
        headers = {'Content-Type': 'application/json'}
        response = requests.post(url, data=json.dumps(payload), headers=headers)
        
        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            return response.json()  # Return the response as JSON
        else:
            return None  # Return None if the request was unsuccessful
        
    def _decode_b64_numpy(self, b64):
        buffer = base64.b64decode(b64.encode('ascii'))
        arr = np.frombuffer(buffer, dtype=np.float16)
        return arr
import time
import logging
import os

import numpy as np
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)

logger = logging.getLogger(__name__)

fmt = "\n=== {:30} ===\n"
search_latency_fmt = "search latency = {:.4f}s"

class MilvusInterface:
    def __init__(self) -> None:
        logger.info(fmt.format("start connecting to Milvus"))
        connections.connect("default", host="host.docker.internal", port=os.getenv("MILVUS_PORT"))
        
        # define schema:
        dim = int(os.getenv("EMBEDDING_DIM"))
        fields = [
            FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="source_file", dtype=DataType.VARCHAR, max_length=300),
            FieldSchema(name="text_preview", dtype=DataType.VARCHAR, max_length=300),
            FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=dim)
        ]
        schema = CollectionSchema(fields, "documents holds document embeddings")

        logger.info(fmt.format("Create collection `documents`"))
        # if the schema is changed, remove the old one with utility.drop_collection("documents")
        self.documents = Collection("documents", schema, consistency_level="Strong")
        logger.info(fmt.format("Making L2 index for `documents`"))
        self.make_index()
        self.needs_load = True

    def insert(self, source_paths: list, text_preview: list, embeddings: list):
        text_preview = [t[:300] for t in text_preview]
        entities = [
            source_paths,
            text_preview,
            embeddings
        ]
        insert_result = self.documents.insert(entities)
        self.needs_load = True

    def insert_bulk(self, source_paths: list, embeddings: list):
        entities = [
            source_paths,
            embeddings
        ]
        insert_result = self.documents.insert(entities)
        '''When data is inserted into Milvus it is inserted into segments. 
        Segments have to reach a certain size to be sealed and indexed. 
        Unsealed segments will be searched brute force. In order to avoid this with any 
        remainder data, it is best to call flush().
        The flush call will seal any remaining segments and send them for indexing. 
        It is important to only call this at the end of an insert session, as calling this 
        too much will cause fragmented data that will need to be cleaned later on.'''
        self.documents.flush()
        logger.info(f"Number of entities in Milvus: {self.documents.num_entities}")
        
    def query(self, embedding, limit=10):
        if self.needs_load:
            # prepare for querying by loading into memory
            self.documents.load()
            self.needs_load = False

        search_params = {
            "metric_type": "L2",
            "params": {"nprobe": 10},
        }

        start_time = time.time()
        result = self.documents.search([embedding], "embeddings", search_params, limit=limit, output_fields=["source_file", "text_preview"])
        end_time = time.time()

        logger.info(search_latency_fmt.format(end_time - start_time))

        formatted_results = []
        for hits in result:
            for hit in hits:
                logger.info(f"hit: {hit}, source_file field: {hit.entity.get('source_file')}")
                formatted_results.append({"source_file":hit.entity.get('source_file'), "text_preview":hit.entity.get('text_preview')})
        return formatted_results
    
    def make_index(self):
        index = {
            "index_type": "IVF_FLAT",
            "metric_type": "L2",
            "params": {"nlist": 128},
        }

        self.documents.create_index("embeddings", index)

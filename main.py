from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from milvus import default_server
from pymilvus import Collection, utility
from sentence_transformers import SentenceTransformer

from conf import settings
from helpers.VectorMaker import VectorMaker
from helpers.helpers import start_server, generate_doc_map
from schemas.responses import QuestionResponse

ml_resources = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """This function is executed at startup.
    Start milvus server and populate the ml_resources dictionary.
    These resources are to be used by the whole app, and are shared among requests
    """
    print('starting milvus server')
    # start milvus
    default_server.set_base_dir(settings.milvus_foldername)
    start_server()
    # collection name can only contain numbers, letters and underscores
    collection_name = settings.inference_model.replace('-', '_')
    if utility.has_collection(collection_name):
        ml_resources['collection'] = Collection(collection_name)
    else:
        print('generating collection')
        vector_maker = VectorMaker(settings.inference_model, settings.inference_model_dim)
        vector_maker.generate_collection()
        ml_resources['collection'] = Collection(collection_name)
    print('Creating ML resources')
    # Load the sentence_transformer model and doc map
    ml_resources["sentence_transformer"] = SentenceTransformer(settings.inference_model)
    ml_resources["justice_doc_map"] = generate_doc_map()
    yield
    # Clean up the ML models and release the resources
    ml_resources.clear()
    default_server.stop()


# create fastapi instance
app = FastAPI(lifespan=lifespan)


@app.get("/chatbot_of_justice/process_question",  response_model=QuestionResponse)
def process_question(question: str):
    search_data = ml_resources['sentence_transformer'].encode([question])
    res = ml_resources['collection'].search(
        data=search_data,  # Embeded search value
        anns_field="facts_embedding",  # Search across embeddings
        param={},
        limit=1,  # Limit to n results per search
        output_fields=['chunked_facts', 'id']  # Include title field in result
    )
    return ({
        'document_id': res[0].ids[0],
        'document_text': ml_resources['justice_doc_map'].get(res[0].ids[0]),
        'relevant_sentence': res[0][0].fields['chunked_facts']
    })


@app.get("/")
async def root():
    return {"msg": "visit /docs for swagger"}


# set parameters to run uvicorn
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="localhost",
        port=8000,
        log_level="info",
        workers=1,
    )

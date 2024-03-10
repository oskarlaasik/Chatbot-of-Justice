import time

from datasets import load_dataset
from milvus import default_server
from pymilvus import connections, Collection
from sentence_transformers import SentenceTransformer

from conf import settings

default_server.set_base_dir(settings.milvus_foldername)
connections.connect(host='127.0.0.1', port=default_server.listen_port)
# collection name can only contain numbers, letters and underscores
collection = Collection(settings.pretrained_model_to_evaluate.replace('-', '_'))
transformer = SentenceTransformer(settings.pretrained_model_to_evaluate)

question_dataset = load_dataset("csv", data_files=settings.question_dataset_path, split='all')
search_terms = question_dataset['question']


# Search the database based on input text
def embed_search(data: list):
    embeds = transformer.encode(data)
    return [x for x in embeds]


search_data = embed_search(search_terms)

start = time.time()
res = collection.search(
    data=search_data,  # Embeded search value
    anns_field="facts_embedding",  # Search across embeddings
    param={},
    limit=settings.result_limit,  # Limit to n results per search
    output_fields=['chunked_facts', 'id']  # Include title field in result
)
end = time.time()

for hits_i, hits in enumerate(res):
    print('facts_embedding:', search_terms[hits_i])
    print('Search Time:', end - start)
    print('Results:')
    for hit in hits:
        print(hit.entity.get('chunked_facts'), '----', hit.distance)
    print()

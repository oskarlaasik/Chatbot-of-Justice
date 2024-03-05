import time

from datasets import load_dataset
from milvus import default_server
from pymilvus import Collection
from pymilvus import connections

from Preprocessor import Preprocessor
from conf import settings

default_server.set_base_dir('test_milvus')

# In case server is running, it is quicker to try to connect
# and in case of failure start server, server startup hangs for 3 min
try:
    connections.connect(host='127.0.0.1', port=default_server.listen_port)
except:
    default_server.start()
    connections.connect(host='127.0.0.1', port=default_server.listen_port)

TOP_K = 1
question_dataset = load_dataset("csv", data_files='data/question_data.csv', split='all')

search_terms = question_dataset['question']


def print_results(res, duration, modelname):
    overall_score = 0
    for hits_i, hits in enumerate(res):
        if question_dataset[hits_i]['document_id'] == res[hits_i].ids[0]:
            overall_score += 1

    score = overall_score/len(res)
    print(modelname + ': ' + str(score))
    print('time' + ': ' + str(duration))


for i in range(len(settings.models_to_test)):
    print(settings.models_to_test[i])
    preprocessor = Preprocessor(settings.models_to_test[i])
    embeds = preprocessor.embed(search_terms)
    collection = Collection(settings.models_to_test[i].replace('-', '_'))
    start = time.time()
    res = collection.search(
        data=embeds,  # Embeded search value
        anns_field="facts_embedding",  # Search across embeddings
        param={},
        limit=TOP_K,  # Limit to top_k results per search
        output_fields=['chunked_facts']  # Include title field in result
    )
    end = time.time()

    duration = end - start
    print_results(res, duration, settings.models_to_test[i])

default_server.stop()

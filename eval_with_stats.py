import time

from datasets import load_dataset
from pymilvus import Collection, SearchResult

from conf import settings
from helpers.Preprocessor import Preprocessor
from helpers.helpers import start_server

start_server()

question_dataset = load_dataset("csv", data_files=settings.question_dataset_path, split='all')
search_terms = question_dataset['question']


def print_results(res: SearchResult, duration: int, modelname: str):
    overall_score = 0
    for hits_i, hits in enumerate(res):
        if question_dataset[hits_i]['document_id'] == res[hits_i].ids[0]:
            overall_score += 1

    score = overall_score / len(res)
    print(modelname + ': ' + str(score))
    print('time' + ': ' + str(duration))


for i in range(len(settings.models_to_test)):
    print(settings.models_to_test[i])
    preprocessor = Preprocessor(settings.models_to_test[i])
    embeds = preprocessor.embed(search_terms)
    # collection name can only contain numbers, letters and underscores
    collection = Collection(settings.models_to_test[i].replace('-', '_'))
    start = time.time()
    res = collection.search(
        data=embeds,  # Embeded search value
        anns_field="facts_embedding",  # Search across embeddings
        param={},
        limit=settings.result_limit,  # Limit to 1 results per search
        output_fields=['chunked_facts']  # Include title field in result
    )
    end = time.time()

    duration = end - start
    print_results(res, duration, settings.models_to_test[i])

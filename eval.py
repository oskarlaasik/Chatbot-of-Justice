from datasets import load_dataset
from pymilvus import Collection
from milvus import default_server
from pymilvus import connections
from Preprocessor import Preprocessor
from conf import settings


default_server.set_base_dir('test_milvus')
if default_server.running is False:
    default_server.start()

# Now you could connect with localhost and the given port
# Port is defined by default_server.listen_port
connections.connect(host='127.0.0.1', port=default_server.listen_port)

preprocessor = Preprocessor('bert-base-uncased')

question_dataset = load_dataset("csv", data_files='data/question_data.csv', split='all')
question_dataset = question_dataset.map(preprocessor.tokenize, batched=True,
                                        batch_size=settings.TOKENIZATION_BATCH_SIZE,
                                        fn_kwargs={'column_name': 'question'})
question_dataset.set_format('torch', columns=['input_ids', 'token_type_ids', 'attention_mask'], output_all_columns=True)
question_dataset = question_dataset.map(preprocessor.embed,
                                        remove_columns=['input_ids', 'token_type_ids', 'attention_mask'],
                                        batched=True, batch_size=settings.INFERENCE_BATCH_SIZE,
                                        fn_kwargs={'embedding_name': 'question_embedding'})

collection = Collection("bert_base_uncased")


def search(batch):
    res = collection.search(batch['question_embedding'].tolist(), anns_field='facts_embedding', param={},
                            output_fields=['chunks'], limit=settings.LIMIT)
    overall_id = []
    overall_distance = []
    overall_answer = []
    for hits in res:
        ids = []
        distance = []
        answer = []
        for hit in hits:
            ids.append(hit.id)
            distance.append(hit.distance)
            answer.append(hit.entity.get('chunks'))
        overall_id.append(ids)
        overall_distance.append(distance)
        overall_answer.append(answer)
    return {
        'id': overall_id,
        'distance': overall_distance,
        'chunks': overall_answer,
    }


question_dataset = question_dataset.map(search, batched=True, batch_size=1)

for x in question_dataset:
    print()
    print('Question:')
    print(x['question'])
    print('Answer, Distance')
    for x in zip(x['chunks'], x['distance']):
        print(x)

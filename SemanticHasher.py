from datasets import load_dataset
from pymilvus import FieldSchema, CollectionSchema, DataType, Collection, utility

from Preprocessor import Preprocessor
from conf import settings


class SemanticHasher:
    def __init__(self, collection_name):
        self.collection = self.create_collection(collection_name)

    def insert_function(self, batch):
        insertable = [
            batch['id'].tolist(),
            batch['chunks'],
            batch['facts_embedding'].tolist()
        ]
        self.collection.insert(insertable)

    def create_collection(self, collection_name):
        if utility.has_collection(collection_name):
            utility.drop_collection(collection_name)

        index_params = {
            'metric_type': 'IP',
            'index_type': "IVF_FLAT",
            'params': {"nlist": 1536}
        }

        fields = [
            FieldSchema(name='id', dtype=DataType.INT64, is_primary=True),
            FieldSchema(name='chunks', dtype=DataType.VARCHAR, max_length=settings.TOKENIZATION_BATCH_SIZE),
            FieldSchema(name='facts_embedding', dtype=DataType.FLOAT_VECTOR, dim=settings.DIMENSION)
        ]
        schema = CollectionSchema(fields=fields)
        collection = Collection(name=collection_name, schema=schema)

        collection.create_index(field_name="facts_embedding", index_params=index_params)
        collection.load()
        return collection

    def hash_collection(self, model_name):
        preprocessor = Preprocessor(model_name)
        print('loading dataset from csv')
        dataset = load_dataset("csv", data_files='data/justice.csv', split='all')
        print('chunking')
        dataset = dataset.map(preprocessor.chunk_examples, batch_size=16, batched=True,
                              remove_columns=dataset.column_names)
        print('tokenizing')
        dataset = dataset.map(preprocessor.tokenize, batch_size=settings.TOKENIZATION_BATCH_SIZE, batched=True,
                              fn_kwargs={'column_name': 'chunks'})
        print('calculating embeddings')
        dataset.set_format('torch', columns=['input_ids', 'token_type_ids', 'attention_mask'],
                           output_all_columns=True)
        dataset = dataset.map(preprocessor.embed, remove_columns=['input_ids', 'token_type_ids', 'attention_mask'],
                              batched=True,
                              batch_size=settings.INFERENCE_BATCH_SIZE,
                              fn_kwargs={'embedding_name': 'facts_embedding'})
        print('loading data to vector database')
        dataset.map(self.insert_function, batched=True, batch_size=64)
        self.collection.flush()

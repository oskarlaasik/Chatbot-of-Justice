from milvus import default_server
from pymilvus import utility


class SemanticHasher:
    def hash_collection(self, collection_name):
        if utility.has_collection(COLLECTION_NAME):
            utility.drop_collection(COLLECTION_NAME)


default_server.set_base_dir('test_milvus')
# default_server.start()

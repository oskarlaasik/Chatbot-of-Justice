from milvus import default_server
from pymilvus import connections

from SemanticHasher import SemanticHasher

default_server.set_base_dir('test_milvus')
#if default_server.running is False:
#    default_server.start()

# Now you could connect with localhost and the given port
# Port is defined by default_server.listen_port
connections.connect(host='127.0.0.1', port=default_server.listen_port)

sem_hasher = SemanticHasher('bert_base_uncased')
sem_hasher.hash_collection('bert-base-uncased')

default_server.stop()

from milvus import default_server
from pymilvus import connections

from SemanticHasher import SemanticHasher
from conf import settings


default_server.set_base_dir('test_milvus')

# In case server is running, it is quicker to try to connect
# and in case of failure start server, server startup hangs for 3 min
try:
    connections.connect(host='127.0.0.1', port=default_server.listen_port)
except:
    default_server.start()
    connections.connect(host='127.0.0.1', port=default_server.listen_port)


for i in range(len(settings.models_to_test)):
    print(settings.models_to_test[i])
    sem_hasher = SemanticHasher(settings.models_to_test[i], settings.corresponding_model_dim[i])
    sem_hasher.hash_collection(settings.models_to_test[i])

default_server.stop()

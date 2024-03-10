from milvus import default_server

from conf import settings
from helpers.VectorMaker import VectorMaker
from helpers.helpers import start_server

default_server.set_base_dir(settings.milvus_foldername)
start_server()

for i in range(len(settings.models_to_test)):
    print(settings.models_to_test[i])
    vector_maker = VectorMaker(settings.models_to_test[i], settings.corresponding_model_dim[i])
    vector_maker.generate_collection()

import csv

from milvus import default_server
from pymilvus import connections

from conf import settings


def start_server():
    default_server.set_base_dir(settings.milvus_foldername)
    # In case server is running, it is quicker to try to connect
    # and in case of failure start server, server startup hangs for 3 min
    try:
        connections.connect(host='127.0.0.1', port=default_server.listen_port)
    except Exception as e:
        default_server.start()
        connections.connect(host='127.0.0.1', port=default_server.listen_port)



def generate_doc_map():
    # Generate dict for connecting document id's to text
    with open(settings.justice_dataset_path, 'r') as file:
        doc_map = {}
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            doc_map[int(row['ID'])] = row['facts']
    return doc_map

default_server.set_base_dir('test_milvus')
if default_server.running == False:
    default_server.start()
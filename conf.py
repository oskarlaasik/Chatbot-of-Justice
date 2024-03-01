from pydantic import BaseSettings


class Settings(BaseSettings):
    # loend mudelitest, mille efektiivsust oleks vaja testida
    models_to_test = ['bert-base-uncased', 'bert-large-cased']

    TOKENIZATION_BATCH_SIZE = 1024  # Batch size for tokenizing operation
    INFERENCE_BATCH_SIZE = 8  # batch size for transformer
    INSERT_RATIO = .01  # How many titles to embed and insert
    COLLECTION_NAME = 'new_data_test'  # Collection name
    DIMENSION = 768  # Embeddings size
    DIMENSION = 1024  # Embeddings size
    LIMIT = 5  # How many results to search for


settings = Settings()

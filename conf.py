from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # loend mudelitest, mille efektiivsust oleks vaja testida
    models_to_test: list = ['bert-base-uncased', 'bert-large-cased']

    TOKENIZATION_BATCH_SIZE: int = 1024  # Batch size for tokenizing operation
    INFERENCE_BATCH_SIZE: int = 8  # batch size for transformer
    COLLECTION_NAME: str = 'new_data_test'  # Collection name
    DIMENSION: int = 768  # Embeddings size
    LIMIT: int = 5  # How many results to search for


settings = Settings()

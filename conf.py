from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # loend mudelitest, mille efektiivsust oleks vaja testida
    models_to_test: list = [
        'all-mpnet-base-v2',
        'multi-qa-mpnet-base-dot-v1',
        'multi-qa-distilbert-cos-v1',
        'multi-qa-MiniLM-L6-cos-v1',
        'all-distilroberta-v1',
        'all-MiniLM-L12-v2',
        'all-MiniLM-L6-v2',
        'paraphrase-multilingual-mpnet-base-v2',
        'paraphrase-albert-small-v2',
        'paraphrase-MiniLM-L3-v2',
        'paraphrase-multilingual-MiniLM-L12-v2',
        'distiluse-base-multilingual-cased-v1',
        'distiluse-base-multilingual-cased-v2',
        'multi-qa-mpnet-base-cos-v1',
        'multi-qa-mpnet-base-dot-v1',
        'multi-qa-mpnet-base-dot-v1'

    ]


    corresponding_model_dim: list = [
        768,
        768,
        768,
        384,
        768,
        384,
        384,
        768,
        768,
        384,
        384,
        512,
        512,
        768,
        768
    ]
    TOKENIZATION_BATCH_SIZE: int = 1024  # Batch size for tokenizing operation
    INFERENCE_BATCH_SIZE: int = 8  # batch size for transformer
    LIMIT: int = 5  # How many results to search for


settings = Settings()

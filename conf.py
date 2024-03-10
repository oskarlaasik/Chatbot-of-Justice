from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # models to test when looking at stats
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
        'msmarco-bert-base-dot-v5'
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
        768,
        768
    ]
    tokenization_batch_size: int = 1024  # Batch size for tokenizing operation
    pretrained_model_to_evaluate: str = 'all-mpnet-base-v2'  # When evaluating a single model
    result_limit: int = 5  # How many results to search for when evaluating a model
    inference_model: str = 'all-mpnet-base-v2'
    inference_model_dim: int = 768
    milvus_foldername: str = 'milvus_data'
    question_dataset_path: str = 'data/question_data.csv'
    justice_dataset_path: str = 'data/justice.csv'


settings = Settings()

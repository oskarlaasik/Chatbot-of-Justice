import pickle
import re

from sentence_transformers import SentenceTransformer


class Preprocessor:
    def __init__(self, model_name: str):
        with open('data/english.pickle', 'rb') as punkt_file:
            self.sentence_tokenizer = pickle.load(punkt_file)
        self.model_name = model_name
        self.transformer = SentenceTransformer(model_name)

    def chunk_examples(self, batch):
        """Chunk incoming text batches into paragraphs and paragraphs into sentences.
        Add original document id connecting sentences to dataset columns.
        """
        chunks = []
        ids = []
        for i in range(len(batch['facts'])):
            paragraphs = re.split('</p>|<p>', batch['facts'][i].strip())
            paragraphs = list(filter(lambda sentence: sentence.strip(), paragraphs))
            sentences = []
            for paragraph in paragraphs:
                sent_text = self.sentence_tokenizer.tokenize(paragraph)
                sentences.extend(sent_text)
            chunks.extend(sentences)
            ids.extend([batch['ID'][i]] * len(sentences))
        return {'chunked_facts': chunks, 'id': ids}

    def embed(self, data: list):
        """use SentenceTransformers library o compute embeddings for a list ofsentences.
        """
        return self.transformer.encode(data)


import re

import nltk
from torch import clamp, sum
from transformers import AutoTokenizer


class Preprocessor:
    def __init__(self, model):
        self.tokenizer = AutoTokenizer.from_pretrained(model)

    def chunk_examples(self, batch):
        chunks = []
        ids = []
        for i in range(len(batch['facts'])):
            paragraphs = re.split('</p>|<p>', batch['facts'][i].strip())
            paragraphs = list(filter(lambda name: name.strip(), paragraphs))
            sentences = []
            for paragraph in paragraphs:
                sent_text = nltk.sent_tokenize(paragraph)
                sentences.extend(sent_text)
            chunks.extend(sentences)
            ids.extend([batch['ID'][i]] * len(sentences))
        return {'chunks': chunks, 'id': ids}

    def tokenize(self, batch):
        results = self.tokenizer(batch['chunks'], add_special_tokens=True, truncation=True, padding="max_length",
                                 return_attention_mask=True, return_tensors="pt")
        batch['input_ids'] = results['input_ids']
        batch['token_type_ids'] = results['token_type_ids']
        batch['attention_mask'] = results['attention_mask']
        return batch

    def embed_question(self, batch, embedding_name):
        sentence_embs = self.model(
            input_ids=batch['input_ids'],
            token_type_ids=batch['token_type_ids'],
            attention_mask=batch['attention_mask']
        )[0]
        input_mask_expanded = batch['attention_mask'].unsqueeze(-1).expand(sentence_embs.size()).float()
        batch[embedding_name] = sum(sentence_embs * input_mask_expanded, 1) / clamp(input_mask_expanded.sum(1),
                                                                                    min=1e-9)
        return batch

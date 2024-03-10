from datasets import load_dataset
import numpy as np
from helpers.Preprocessor import Preprocessor


class TestPreprocessor:
    dataset = load_dataset("csv", data_files='tests/payloads/test_data.csv', split='all')
    # Give random model name to initiate preprocessor
    preprocessor = Preprocessor('paraphrase-MiniLM-L3-v2')

    def test_tokenizer(self):
        dataset = self.dataset.map(self.preprocessor.chunk_examples, batch_size=16, batched=True,
                                   remove_columns=self.dataset.column_names)
        assert len(dataset['chunked_facts']) == 3
        assert dataset['chunked_facts'][2] == 'Look here, a third one.'
        assert dataset['id'] == [1, 1, 1]

    def test_embed(self):
        result = self.preprocessor.embed('This is a sentence')
        assert type(result) == np.ndarray

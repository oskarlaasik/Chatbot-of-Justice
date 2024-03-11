# Chatbot of Justice

Chatbot of Justice is a python library to vectorize and retrieve supreme court dataset.

## Usage

The library uses the supreme court dataset(https://www.kaggle.com/datasets/deepcontractor/supreme-court-judgment-prediction) and assumes it is in the data folder: data/justice.csv.

## Running locally

Install dependancies

```bash
pip install -r requirements.txt
```
### Populating vector database
To evaluate all the models defined in the conf.py configuration file, you need to first run:
```bash
python make_and_insert_embs.py
```
### Evaluating all the models

There are 33 questions in the data/question_data.csv file. All models are evaluated using this questionnaire measuring it's ability to match the questions to the dataset.

To see the score and time for each model. 

```bash
python eval_with_stats.py
```

To see specific predictions each model:

```bash
python eval_single_model.py
```

### Serving the model with FastAPI

```bash
python main.py
```
By default library runs on port 8000. Be patient, starting the Milvus server takes a moment.

go to http://localhost:8000/docs to use swagger and in the /chatbot_of_justice/process_question subsection write your question to  into the corresponding field.

### Running tests

Please make sure you are running tests from the project root directory.


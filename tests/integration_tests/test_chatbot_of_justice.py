import json

from fastapi.testclient import TestClient

from main import app


class TestChatbotOfJustice:

    def test_process_question(self):
        with TestClient(app) as client:
            response = client.get(
                "/chatbot_of_justice/process_question?question=Is it legal to drink beer and drive a vehicle in Kansas?")
        assert response.status_code == 200
        assert 'document_id' in response.json()
        assert 'document_text' in response.json()
        assert 'relevant_sentence' in response.json()


    def test_process_root(self):
        with TestClient(app) as client:
            response = client.get("/")
        assert response.status_code == 200
        assert response.json() == {"msg": "visit /docs for swagger"}

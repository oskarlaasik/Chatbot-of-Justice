from pydantic import BaseModel


class QuestionResponse(BaseModel):
    document_id: int
    document_text: str
    relevant_sentence: str

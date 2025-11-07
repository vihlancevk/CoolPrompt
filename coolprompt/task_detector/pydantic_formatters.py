from pydantic import BaseModel, Field


class TaskDetectionStructuredOutputSchema(BaseModel):
    task: str = Field(description="Determined task classification")

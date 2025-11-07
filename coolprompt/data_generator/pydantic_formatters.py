from pydantic import BaseModel, Field


class ProblemDescriptionStructuredOutputSchema(BaseModel):
    problem_description: str = Field(description="Determined problem description")


class ClassificationTaskExample(BaseModel):
    input: str = Field(description="Input request")
    output: str = Field(description="Output label")


class ClassificationTaskStructuredOutputSchema(BaseModel):
    examples: list[ClassificationTaskExample] = Field(
        description="List of examples like " + '{"input": "...", "output": "ground-truth label"}'
    )


class GenerationTaskExample(BaseModel):
    input: str = Field(description="Input request")
    output: str = Field(description="LLM answer")


class GenerationTaskStructuredOutputSchema(BaseModel):
    examples: list[GenerationTaskExample] = Field(description='List of examples like {"input": "...", "output": "..."}')

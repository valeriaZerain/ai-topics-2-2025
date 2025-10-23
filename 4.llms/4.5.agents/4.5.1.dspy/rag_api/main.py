from fastapi import FastAPI
from typing import Any
from agent import RAG

app = FastAPI(title="Los Kjarkas AI Agent")

rag = RAG()


@app.post("/qa")
def query_agent(
    question: str,
    user_id: int,
) -> dict[str, Any]:
    answer = rag(question)

    return {"answer": answer, "status": "success"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app)
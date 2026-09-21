from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel

from src.llm import load_rag_components, ask_rag


app = FastAPI(
    title="PDF RAG Q&A API",
    description="RAG-based PDF question answering API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading RAG system...")

components = load_rag_components()

print("RAG system ready.")


class QuestionRequest(BaseModel):
    query: str


class Source(BaseModel):
    chunk_id: int
    pages: list
    section_path: list


class QuestionResponse(BaseModel):
    answer: str
    sources: list[Source]


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "message": "RAG API is running",
    }


@app.post("/ask", response_model=QuestionResponse)
def ask_question(request: QuestionRequest):

    result = ask_rag(
        components=components,
        query=request.query,
    )

    sources = []

    for source in result["sources"]:

        metadata = source["document"].metadata

        sources.append(
            {
                "chunk_id": int(metadata.get("chunk_id")),
                "pages": metadata.get("pages", []),
                "section_path": metadata.get(
                    "section_path",
                    [],
                ),
            }
        )

    return {
        "answer": result["answer"],
        "sources": sources,
    }
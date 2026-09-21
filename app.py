import os
from pathlib import Path
import uvicorn

# Import existing FastAPI application
from api.main import app, components
from src.llm import ask_rag

# Optional: Mount a native Gradio interface at /gradio for Hugging Face Spaces compatibility
try:
    import gradio as gr

    def gradio_ask(question: str):
        if not question or not question.strip():
            return "Please enter a question.", ""

        result = ask_rag(components=components, query=question)
        answer = result["answer"]

        sources_text = ""
        for i, source in enumerate(result["sources"], start=1):
            metadata = source["document"].metadata
            chunk_id = metadata.get("chunk_id", "N/A")
            pages = metadata.get("pages", [])
            section = metadata.get("section_path", [])

            pages_str = ", ".join(str(p) for p in pages) if isinstance(pages, list) else str(pages)
            sec_str = " > ".join(str(s) for s in section) if isinstance(section, list) else str(section)

            sources_text += f"\n- **Source {i}** | **Chunk:** `{chunk_id}` | **Page:** `{pages_str}` | **Section:** `{sec_str}`\n"

        return answer, sources_text

    with gr.Blocks(title="PDF RAG Q&A") as demo:
        gr.Markdown("# 📄 PDF RAG Q&A Chatbot")
        gr.Markdown(
            "Ask questions grounded in the provided document with verified source citations.\n\n"
            "*(Note: You can also use the custom web UI directly at the root `/` URL!)*"
        )
        with gr.Row():
            with gr.Column():
                query_input = gr.Textbox(
                    lines=3,
                    label="Your Question",
                    placeholder="e.g. What is positional encoding and why is it needed?"
                )
                submit_btn = gr.Button("Ask Question", variant="primary")
            with gr.Column():
                answer_output = gr.Markdown(label="Generated Answer")
                sources_output = gr.Markdown(label="Referenced Sources")

        submit_btn.click(
            fn=gradio_ask,
            inputs=query_input,
            outputs=[answer_output, sources_output]
        )

    # Mount Gradio at /gradio; the custom frontend is served at /
    app = gr.mount_gradio_app(app, demo, path="/gradio")

except Exception as err:
    print(f"Gradio mounting notice: {err}")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    print(f"Starting server on 0.0.0.0:{port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)

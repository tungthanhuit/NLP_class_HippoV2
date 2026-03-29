import os
from typing import Optional, Tuple, List, Dict

import gradio as gr

from src.hipporag import HippoRAG


CfgTuple = Tuple[str, str, str, str, str]
# (save_dir, llm_model_name, base_url, embedding_model_name, embedding_trust_remote_code)


def _normalize_base_url(value: str) -> Optional[str]:
    value = (value or "").strip()
    return value or None


def _parse_docs(text: str) -> List[str]:
    lines = (text or "").splitlines()
    docs = [ln.strip() for ln in lines]
    return [d for d in docs if d]


def get_or_init_rag(
    rag: Optional[HippoRAG],
    cfg_state: Optional[CfgTuple],
    save_dir: str,
    llm_model_name: str,
    base_url: str,
    embedding_model_name: str,
    embedding_trust_remote_code: bool,
) -> Tuple[HippoRAG, CfgTuple]:
    save_dir = (save_dir or "outputs/gradio_demo").strip()
    llm_model_name = (llm_model_name or "gpt-4o-mini").strip()
    embedding_model_name = (embedding_model_name or "text-embedding-3-small").strip()

    base_url_norm = _normalize_base_url(base_url)
    cfg: CfgTuple = (
        save_dir,
        llm_model_name,
        base_url_norm or "",
        embedding_model_name,
        str(bool(embedding_trust_remote_code)),
    )

    if rag is None or cfg_state != cfg:
        rag = HippoRAG(
            save_dir=save_dir,
            llm_model_name=llm_model_name,
            llm_base_url=base_url_norm,
            embedding_model_name=embedding_model_name,
            embedding_base_url=base_url_norm,
            embedding_trust_remote_code=embedding_trust_remote_code,
        )
        cfg_state = cfg
    elif cfg_state is None:
        # Defensive: keep state consistent even if only rag was passed.
        cfg_state = cfg

    return rag, cfg_state


def update_kb(
    save_dir: str,
    llm_model_name: str,
    base_url: str,
    embedding_model_name: str,
    embedding_trust_remote_code: bool,
    docs_text: str,
    rag_state: Optional[HippoRAG],
    cfg_state: Optional[CfgTuple],
) -> Tuple[Optional[HippoRAG], Optional[CfgTuple], str]:
    docs = _parse_docs(docs_text)
    if not docs:
        return rag_state, cfg_state, "No docs provided. Paste one document per line."

    try:
        rag, cfg_state = get_or_init_rag(
            rag_state,
            cfg_state,
            save_dir,
            llm_model_name,
            base_url,
            embedding_model_name,
            embedding_trust_remote_code,
        )

        before = len(rag.chunk_embedding_store.get_all_texts())
        rag.index(docs=docs)
        after = len(rag.chunk_embedding_store.get_all_texts())

        base_url_norm = _normalize_base_url(base_url)
        resolved_base = base_url_norm or "(OpenAI default)"

        status = (
            "KB update complete.\n\n"
            f"- Added input docs: {len(docs)}\n"
            f"- Stored docs before/after: {before} → {after} (Δ {after - before})\n"
            f"- save_dir: {rag.global_config.save_dir}\n"
            f"- working_dir: {rag.working_dir}\n"
            f"- llm: {rag.global_config.llm_name} @ {resolved_base}\n"
            f"- embedding: {rag.global_config.embedding_model_name}\n"
        )
        return rag, cfg_state, status
    except Exception as e:
        return rag_state, cfg_state, f"Update KB failed: {type(e).__name__}: {e}"


def chat_send(
    message: str,
    history: List[Dict[str, str]],
    save_dir: str,
    llm_model_name: str,
    base_url: str,
    embedding_model_name: str,
    embedding_trust_remote_code: bool,
    rag_state: Optional[HippoRAG],
    cfg_state: Optional[CfgTuple],
) -> Tuple[List[Dict[str, str]], str, Optional[HippoRAG], Optional[CfgTuple], str]:
    message = (message or "").strip()
    if not message:
        return history, "", rag_state, cfg_state, ""

    try:
        rag, cfg_state = get_or_init_rag(
            rag_state,
            cfg_state,
            save_dir,
            llm_model_name,
            base_url,
            embedding_model_name,
            embedding_trust_remote_code,
        )

        if len(rag.chunk_embedding_store.get_all_texts()) == 0:
            answer = "KB is empty. Go to the 'Add docs and update KB' tab and add some documents first."
            history = (history or []) + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": answer},
            ]
            return history, "", rag, cfg_state, ""

        result = rag.rag_qa([message])
        solutions = result[0]
        answer = solutions[0].answer
        history = (history or []) + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": answer},
        ]
        return history, "", rag, cfg_state, ""
    except Exception as e:
        err = f"Chat failed: {type(e).__name__}: {e}"
        history = (history or []) + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": err},
        ]
        return history, "", rag_state, cfg_state, err


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="HippoRAG Gradio Demo") as demo:
        rag_state = gr.State(value=None)
        cfg_state = gr.State(value=None)

        gr.Markdown("# HippoRAG Demo\nThree pages: Chat, KB, Settings.")

        # Define settings components up-front so we can wire callbacks,
        # but render them later inside the Settings tab to keep tab order.
        save_dir = gr.Textbox(label="save_dir", value="outputs/gradio_demo", render=False)
        llm_model_name = gr.Textbox(label="llm_model_name", value="gpt-4o-mini", render=False)
        base_url = gr.Textbox(
            label="base_url (optional)",
            value="http://localhost:4000/v1",
            placeholder="Example: http://localhost:4000/v1 (leave empty for OpenAI)",
            render=False,
        )
        embedding_model_name = gr.Textbox(
            label="embedding_model_name",
            value="text-embedding-3-small",
            placeholder="e.g., text-embedding-3-small or Transformers/sentence-transformers/all-MiniLM-L6-v2",
            render=False,
        )
        embedding_trust_remote_code = gr.Checkbox(
            label="embedding_trust_remote_code",
            value=False,
            render=False,
        )

        with gr.Tab("Chat"):
            chatbot = gr.Chatbot(label="Chat", type="messages")
            chat_status = gr.Textbox(label="Status", lines=2)
            with gr.Row():
                message = gr.Textbox(label="Question", placeholder="Ask a question...")
                send_btn = gr.Button("Send")

            send_btn.click(
                fn=chat_send,
                inputs=[
                    message,
                    chatbot,
                    save_dir,
                    llm_model_name,
                    base_url,
                    embedding_model_name,
                    embedding_trust_remote_code,
                    rag_state,
                    cfg_state,
                ],
                outputs=[chatbot, message, rag_state, cfg_state, chat_status],
            )
            message.submit(
                fn=chat_send,
                inputs=[
                    message,
                    chatbot,
                    save_dir,
                    llm_model_name,
                    base_url,
                    embedding_model_name,
                    embedding_trust_remote_code,
                    rag_state,
                    cfg_state,
                ],
                outputs=[chatbot, message, rag_state, cfg_state, chat_status],
            )

        with gr.Tab("KB"):
            docs_text = gr.Textbox(
                label="Docs (one document per line)",
                lines=12,
                placeholder="Paste one document per line...",
            )
            update_btn = gr.Button("Update KB")
            kb_status = gr.Textbox(label="Status", lines=8)

            update_btn.click(
                fn=update_kb,
                inputs=[
                    save_dir,
                    llm_model_name,
                    base_url,
                    embedding_model_name,
                    embedding_trust_remote_code,
                    docs_text,
                    rag_state,
                    cfg_state,
                ],
                outputs=[rag_state, cfg_state, kb_status],
            )

        with gr.Tab("Settings"):
            gr.Markdown("Configure storage + model endpoints used by Chat and KB.")
            with gr.Row():
                save_dir.render()
                llm_model_name.render()
            with gr.Row():
                base_url.render()
                embedding_model_name.render()
                embedding_trust_remote_code.render()

    return demo


def main() -> None:
    demo = build_ui()

    server_name = os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0")
    server_port = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))

    demo.launch(server_name=server_name, server_port=server_port)


if __name__ == "__main__":
    main()

import os
from typing import Optional, Tuple, List, Dict
from dotenv import load_dotenv

import base64
import json

import gradio as gr

from src.hipporag import HippoRAG

load_dotenv()
OEPNAI_API_KEY = os.getenv("OPENAI_API_KEY")

CfgTuple = Tuple[str, str, str, str, str]
# (save_dir, llm_model_name, base_url, embedding_model_name, embedding_trust_remote_code)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
        cfg_state = cfg

    return rag, cfg_state


# ---------------------------------------------------------------------------
# Neo4j graph fetch
# ---------------------------------------------------------------------------

def _fetch_graph_from_neo4j(
    uri: str, user: str, password: str
) -> Tuple[List[dict], List[dict]]:
    """Return (nodes, edges) from Neo4j. Each node/edge is a plain dict."""
    try:
        from neo4j import GraphDatabase
    except ImportError:
        raise RuntimeError("neo4j package not installed. Run: pip install neo4j")

    driver = GraphDatabase.driver(uri, auth=(user, password))
    nodes, edges = [], []

    with driver.session() as session:
        # Fetch all nodes
        result = session.run("MATCH (n) RETURN id(n) AS nid, labels(n) AS labels, properties(n) AS props")
        for rec in result:
            nodes.append({
                "id": rec["nid"],
                "labels": rec["labels"],
                "props": dict(rec["props"]),
            })

        # Fetch all relationships
        result = session.run(
            "MATCH (s)-[r]->(o) "
            "RETURN id(s) AS src, id(o) AS dst, type(r) AS rtype, properties(r) AS props"
        )
        for rec in result:
            edges.append({
                "src": rec["src"],
                "dst": rec["dst"],
                "type": rec["rtype"],
                "props": dict(rec["props"]),
            })

    driver.close()
    return nodes, edges


# ---------------------------------------------------------------------------
# vis.js HTML builder
# ---------------------------------------------------------------------------

_NODE_COLORS = {
    "Entity": {"background": "#4C9BE8", "border": "#2176C7"},
    "Chunk":  {"background": "#F28B30", "border": "#C0621A"},
}
_DEFAULT_COLOR = {"background": "#aaaaaa", "border": "#777777"}


_STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
_VIS_JS = open(os.path.join(_STATIC_DIR, "vis-network.min.js"), encoding="utf-8").read()


def _build_visjs_html(nodes: List[dict], edges: List[dict]) -> str:
    vis_nodes = []
    for n in nodes:
        node_type = n["labels"][0] if n["labels"] else "Unknown"
        color = _NODE_COLORS.get(node_type, _DEFAULT_COLOR)

        if node_type == "Chunk":
            passage  = n["props"].get("passage", "")
            # Short label: first 5 words
            words    = passage.split()
            label    = " ".join(words[:5]) + ("…" if len(words) > 5 else "")
            title    = f"<b>Chunk</b><br>{passage}"
            shape    = "box"
            size     = 14
        else:
            raw   = n["props"].get("name") or str(n["id"])
            label = (raw[:22] + "…") if len(raw) > 22 else raw
            title = f"<b>Entity</b><br>{raw}"
            shape = "dot"
            size  = 20

        vis_nodes.append({
            "id":    n["id"],
            "label": label,
            "title": title,
            # Do NOT set "group" — it overrides per-node color in vis.js
            "color": {
                "background": color["background"],
                "border":     color["border"],
                "highlight":  {"background": color["border"], "border": "#ffffff"},
                "hover":      {"background": color["border"], "border": "#ffffff"},
            },
            "font":  {"color": "#ffffff", "size": 12, "strokeWidth": 2, "strokeColor": "#000000"},
            "shape": shape,
            "size":  size,
        })

    vis_edges = []
    for i, e in enumerate(edges):
        vis_edges.append({
            "id":     i,
            "from":   e["src"],
            "to":     e["dst"],
            "label":  e["type"],
            "arrows": "to",
            "font":   {
                "size": 10, "color": "#eeeeee",
                "strokeWidth": 2, "strokeColor": "#000000",
                "align": "middle",
            },
            "color":  {"color": "#555555", "highlight": "#aaaaaa", "hover": "#aaaaaa"},
            "smooth": {"type": "curvedCW", "roundness": 0.15},
        })

    nodes_json = json.dumps(vis_nodes)
    edges_json = json.dumps(vis_edges)

    # gr.HTML uses innerHTML which strips <script> tags.
    # Wrap in an iframe with base64-encoded srcdoc so scripts execute correctly.
    inner_html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<script>{_VIS_JS}</script>
<style>
  html, body {{ margin:0; padding:0; background:#0f0f1a; overflow:hidden; }}
  #graph {{ width:100%; height:100vh; background:#0f0f1a; }}
  #legend {{
    position:fixed; top:12px; right:16px;
    background:rgba(0,0,0,0.6); border-radius:6px;
    padding:8px 12px; font-family:sans-serif;
  }}
  .legend-item {{ display:flex; align-items:center; gap:8px; margin:4px 0; color:#ddd; font-size:12px; }}
  .dot {{ width:12px; height:12px; border-radius:50%; display:inline-block; }}
  #info {{
    position:fixed; bottom:8px; left:12px;
    color:#666; font-size:11px; font-family:sans-serif;
  }}

  /* vis.js navigation buttons — high contrast override */
  .vis-network .vis-navigation .vis-button {{
    width: 34px !important;
    height: 34px !important;
    background-color: rgba(255,255,255,0.15) !important;
    border: 1.5px solid rgba(255,255,255,0.4) !important;
    border-radius: 6px !important;
    cursor: pointer;
    transition: background-color 0.15s, border-color 0.15s;
  }}
  .vis-network .vis-navigation .vis-button:hover {{
    background-color: rgba(255,255,255,0.35) !important;
    border-color: rgba(255,255,255,0.8) !important;
  }}
  /* Replace default sprite icons with high-contrast Unicode symbols */
  .vis-network .vis-navigation .vis-button.vis-up          {{ background-image:none !important; }}
  .vis-network .vis-navigation .vis-button.vis-down        {{ background-image:none !important; }}
  .vis-network .vis-navigation .vis-button.vis-left        {{ background-image:none !important; }}
  .vis-network .vis-navigation .vis-button.vis-right       {{ background-image:none !important; }}
  .vis-network .vis-navigation .vis-button.vis-zoomIn      {{ background-image:none !important; }}
  .vis-network .vis-navigation .vis-button.vis-zoomOut     {{ background-image:none !important; }}
  .vis-network .vis-navigation .vis-button.vis-zoomExtends {{ background-image:none !important; }}

  .vis-network .vis-navigation .vis-button::after {{
    display:flex; align-items:center; justify-content:center;
    width:100%; height:100%;
    font-size:18px; font-weight:700;
    color:#ffffff; line-height:1;
  }}
  .vis-network .vis-navigation .vis-button.vis-up::after          {{ content:"↑"; }}
  .vis-network .vis-navigation .vis-button.vis-down::after        {{ content:"↓"; }}
  .vis-network .vis-navigation .vis-button.vis-left::after        {{ content:"←"; }}
  .vis-network .vis-navigation .vis-button.vis-right::after       {{ content:"→"; }}
  .vis-network .vis-navigation .vis-button.vis-zoomIn::after      {{ content:"+"; font-size:22px; }}
  .vis-network .vis-navigation .vis-button.vis-zoomOut::after     {{ content:"−"; font-size:22px; }}
  .vis-network .vis-navigation .vis-button.vis-zoomExtends::after {{ content:"⤢"; font-size:16px; }}
</style>
</head>
<body>
<div id="graph"></div>
<div id="legend">
  <div class="legend-item"><span class="dot" style="background:#4C9BE8"></span>Entity</div>
  <div class="legend-item"><span class="dot" style="background:#F28B30;border-radius:2px"></span>Chunk</div>
</div>
<div id="info">Scroll to zoom · Drag to pan · Click = highlight · Double-click = reset</div>
<script>
  const nodesData = new vis.DataSet({nodes_json});
  const edgesData = new vis.DataSet({edges_json});
  const options = {{
    physics: {{
      solver: "forceAtlas2Based",
      forceAtlas2Based: {{
        gravitationalConstant: -80,
        springLength: 160,
        springConstant: 0.06,
        avoidOverlap: 1.0,       // push overlapping nodes apart
      }},
      stabilization: {{ iterations: 300, updateInterval: 30 }},
      minVelocity: 0.5,
    }},
    interaction: {{
      hover: true,
      tooltipDelay: 80,
      navigationButtons: true,
      keyboard: true,
    }},
    nodes: {{
      borderWidth: 2,
      shadow: {{ enabled:true, size:8, color:"rgba(0,0,0,0.4)" }},
      widthConstraint: {{ maximum: 140 }},  // prevent very wide box nodes
    }},
    edges: {{ width:1.5, selectionWidth:3 }},
    layout: {{ improvedLayout: true }},
  }};
  const network = new vis.Network(
    document.getElementById("graph"),
    {{ nodes: nodesData, edges: edgesData }},
    options
  );
  network.on("click", p => {{
    if (!p.nodes.length) {{ nodesData.update(nodesData.getIds().map(id => ({{id, opacity:1}}))); return; }}
    const nid = p.nodes[0];
    const connected = new Set(network.getConnectedNodes(nid));
    nodesData.update(nodesData.getIds().map(id => ({{ id, opacity: connected.has(id)||id===nid ? 1 : 0.12 }})));
  }});
  network.on("doubleClick", () => {{
    nodesData.update(nodesData.getIds().map(id => ({{id, opacity:1}})));
  }});
</script>
</body>
</html>"""

    b64 = base64.b64encode(inner_html.encode("utf-8")).decode("ascii")
    return (
        f'<iframe src="data:text/html;base64,{b64}" '
        f'style="width:100%;height:620px;border:none;border-radius:8px;background:#0f0f1a">'
        f'</iframe>'
    )


# ---------------------------------------------------------------------------
# Gradio callback
# ---------------------------------------------------------------------------

def load_graph(neo4j_uri: str, neo4j_user: str, neo4j_password: str):
    try:
        nodes, edges = _fetch_graph_from_neo4j(neo4j_uri, neo4j_user, neo4j_password)
        if not nodes:
            return "<p style='color:#aaa;padding:20px'>Neo4j is empty. Run the import script first.</p>", \
                   "Neo4j is empty."
        html = _build_visjs_html(nodes, edges)
        return html, f"Loaded {len(nodes)} nodes, {len(edges)} edges."
    except Exception as e:
        msg = f"Failed to load graph: {type(e).__name__}: {e}"
        return f"<p style='color:#f55;padding:20px'>{msg}</p>", msg


# ---------------------------------------------------------------------------
# KB & Chat callbacks
# ---------------------------------------------------------------------------

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
            rag_state, cfg_state, save_dir, llm_model_name,
            base_url, embedding_model_name, embedding_trust_remote_code,
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
            rag_state, cfg_state, save_dir, llm_model_name,
            base_url, embedding_model_name, embedding_trust_remote_code,
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


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def build_ui() -> gr.Blocks:
    with gr.Blocks(title="HippoRAG Gradio Demo") as demo:
        rag_state = gr.State(value=None)
        cfg_state = gr.State(value=None)

        gr.Markdown("# HippoRAG Demo")

        # Settings components — defined here, rendered inside Settings tab
        save_dir = gr.Textbox(label="save_dir", value="outputs/gradio_demo", render=False)
        llm_model_name = gr.Textbox(label="llm_model_name", value="gpt-4o-mini", render=False)
        base_url = gr.Textbox(
            label="base_url (optional)",
            value="https://api.openai.com/v1",
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
            label="embedding_trust_remote_code", value=False, render=False,
        )

        # ── Chat ──────────────────────────────────────────────────────────
        with gr.Tab("Chat"):
            chatbot = gr.Chatbot(label="Chat", type="messages")
            chat_status = gr.Textbox(label="Status", lines=2)
            with gr.Row():
                message = gr.Textbox(label="Question", placeholder="Ask a question...")
                send_btn = gr.Button("Send")

            _chat_inputs = [
                message, chatbot, save_dir, llm_model_name, base_url,
                embedding_model_name, embedding_trust_remote_code, rag_state, cfg_state,
            ]
            _chat_outputs = [chatbot, message, rag_state, cfg_state, chat_status]
            send_btn.click(fn=chat_send, inputs=_chat_inputs, outputs=_chat_outputs)
            message.submit(fn=chat_send, inputs=_chat_inputs, outputs=_chat_outputs)

        # ── KB ────────────────────────────────────────────────────────────
        with gr.Tab("KB"):
            docs_text = gr.Textbox(
                label="Docs (one document per line)", lines=12,
                placeholder="Paste one document per line...",
            )
            update_btn = gr.Button("Update KB")
            kb_status = gr.Textbox(label="Status", lines=8)

            update_btn.click(
                fn=update_kb,
                inputs=[
                    save_dir, llm_model_name, base_url,
                    embedding_model_name, embedding_trust_remote_code,
                    docs_text, rag_state, cfg_state,
                ],
                outputs=[rag_state, cfg_state, kb_status],
            )

        # ── Graph ─────────────────────────────────────────────────────────
        with gr.Tab("Graph"):
            gr.Markdown("### Knowledge Graph from Neo4j")
            with gr.Row():
                neo4j_uri = gr.Textbox(
                    label="Neo4j URI",
                    value=os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
                )
                neo4j_user = gr.Textbox(
                    label="User",
                    value=os.environ.get("NEO4J_USER", "neo4j"),
                )
                neo4j_password = gr.Textbox(
                    label="Password",
                    value=os.environ.get("NEO4J_PASSWORD", "hipporag_secret"),
                    type="password",
                )
            load_btn = gr.Button("Load Graph")
            graph_status = gr.Textbox(label="Status", lines=1)
            graph_plot = gr.HTML(label="Knowledge Graph")

            load_btn.click(
                fn=load_graph,
                inputs=[neo4j_uri, neo4j_user, neo4j_password],
                outputs=[graph_plot, graph_status],
            )

        # ── Settings ──────────────────────────────────────────────────────
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

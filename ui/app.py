"""Streamlit UI: Notes, Ask, Graph, System. Talks only to the HTTP API."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))
from api_client import Api, ApiError  # noqa: E402
from components.evidence_graph import render as render_graph  # noqa: E402
from textutil import highlight, md_escape  # noqa: E402

STATE_LABEL = {
    "ready": "✅ ready",
    "processing": "⏳ processing",
    "failed_update": "⚠️ update failed (previous version is still searchable)",
    "failed": "❌ failed",
}


def show_error(exc: ApiError) -> None:
    st.error(f"{exc.message or exc.code}  \n`{exc.code}` · request `{exc.request_id}`")


def once_key(name: str, api: Api) -> str:
    """Reuse the same idempotency key for a rerun of a pending action."""
    return str(st.session_state.setdefault(name, api.new_key()))


def notes_view(api: Api) -> None:
    st.subheader("Add a note")
    with st.form("create"):
        title = st.text_input("Title")
        text = st.text_area("Text (Turkish or English)", height=150)
        upload = st.file_uploader("…or a .txt / .md file", type=["txt", "md"])
        if st.form_submit_button("Add"):
            try:
                key = once_key("create_key", api)
                if upload is not None:
                    api.upload_note(title, upload.name, upload.getvalue(), key)
                else:
                    api.create_note(title, text, key)
                st.session_state.pop("create_key", None)  # new key only for a new action
                st.success("Queued for indexing.")
            except ApiError as exc:
                show_error(exc)

    st.subheader("Notes")
    try:
        items = api.notes()["items"]
    except ApiError as exc:
        show_error(exc)
        return
    if not items:
        st.info("No notes yet.")
    for n in items:
        job = n["job"] or {}
        with st.expander(f"{n['title']} — {STATE_LABEL.get(n['state'], n['state'])}"):
            if n["state"] == "processing":
                total = max(job.get("chunks_total", 0), 1)
                st.progress(job.get("chunks_done", 0) / total, text=job.get("stage") or "queued")
            if job.get("status") == "FAILED":
                st.warning(f"Indexing failed: {job.get('error_code')}")
                if st.button("Retry", key=f"retry-{n['note_id']}"):
                    try:
                        api.retry_job(job["job_id"])
                        st.rerun()
                    except ApiError as exc:
                        show_error(exc)
            new_text = st.text_area("Replace text", key=f"txt-{n['note_id']}")
            c1, c2 = st.columns(2)
            if c1.button("Replace", key=f"rep-{n['note_id']}"):
                try:
                    api.replace_note(
                        n["note_id"], n["title"], new_text, once_key(f"rk-{n['note_id']}", api)
                    )
                    st.session_state.pop(f"rk-{n['note_id']}", None)
                    st.rerun()
                except ApiError as exc:
                    show_error(exc)
            if c2.button("Delete", key=f"del-{n['note_id']}"):
                try:
                    api.delete_note(n["note_id"])
                    st.rerun()
                except ApiError as exc:
                    show_error(exc)
    st.button("Refresh")


def ask_view(api: Api) -> None:
    q = st.text_input("Question")
    with_graph = st.checkbox("Show supporting subgraph", value=True)
    if st.button("Ask") and q.strip():
        try:
            with st.spinner("Searching your notes…"):
                r = api.ask(q, with_graph)
        except ApiError as exc:
            show_error(exc)
            return
        if r["status"] == "insufficient_evidence":
            st.warning(r["answer"])
        else:
            (st.success if r["status"] == "answered" else st.info)(md_escape(r["answer"]))
        for m in r["missing_information"]:
            st.caption(f"Missing: {m}")
        for c in r["citations"]:
            with st.container(border=True):
                st.markdown(f"**[{c['evidence_id']}] {md_escape(c['title'])}**")
                try:
                    rev = api.revision(c["note_id"], c["revision_id"])
                    st.markdown(highlight(rev["text"], c["start_char"], c["end_char"]))
                except ApiError:
                    st.markdown(md_escape(c["quote"]))
        if r["graph"]["edges"]:
            render_graph(r["graph"])
        if r["diagnostics"].get("limited_by_budget"):
            st.caption("Results were limited by the context budget.")


def graph_view(api: Api) -> None:
    q = st.text_input("Search entity")
    try:
        ents = api.entities(q or None)["items"]
    except ApiError as exc:
        show_error(exc)
        return
    if not ents:
        st.info("No entities with active source support.")
        return
    ent = st.selectbox("Entity", ents, format_func=lambda e: f"{e['name']} ({e['claim_count']})")
    hops = st.radio("Hops", [1, 2], horizontal=True)
    g = api.graph(ent["entity_id"], hops)
    if g["limited"]:
        st.caption("Results limited — narrow the entity selection.")
    render_graph(g)
    edges = g["edges"]
    if edges:
        e = st.selectbox(
            "Edge",
            edges,
            format_func=lambda x: f"{x['predicate']} ({x['polarity']}, {x['modality']})",
        )
        ev = e["evidence"]
        rev = api.revision(ev["note_id"], ev["revision_id"])
        st.markdown(f"**{md_escape(ev['title'])}**")
        st.markdown(highlight(rev["text"], ev["start_char"], ev["end_char"]))


def system_view(api: Api) -> None:
    try:
        st.json(api.ready())
        st.json(api.status())
    except ApiError as exc:
        show_error(exc)


def main() -> None:
    st.set_page_config(page_title="Notes GraphRAG", layout="wide")
    st.title("Notes → Knowledge Graph")
    api = Api()
    tabs = st.tabs(["Notes", "Ask", "Graph", "System"])
    for tab, view in zip(tabs, (notes_view, ask_view, graph_view, system_view), strict=True):
        with tab:
            view(api)


if __name__ == "__main__":
    main()

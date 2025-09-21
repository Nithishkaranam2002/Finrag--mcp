# ui/app.py
from __future__ import annotations

import os
from pathlib import Path

import requests
import streamlit as st
from dotenv import load_dotenv

# ---------- Config ----------
load_dotenv("env/.env")
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/chat")

ROLES = ["EMPLOYEE", "ENGINEERING", "FINANCE", "HR", "MARKETING", "C_LEVEL"]

st.set_page_config(page_title="FinRAG RBAC Chat", layout="centered")
st.title("FinRAG RBAC Chat")

# ---------- Controls ----------
role = st.selectbox("Role", ROLES, index=0)
query = st.text_input("Ask a question about your company docs…")

col1, col2, col3 = st.columns([1, 1, 1])
top_k = col1.number_input("Top results (k)", min_value=1, max_value=200, value=50, step=1, help="How many chunks to retrieve and display.")
ctx_k = col2.number_input("Context chunks (ctx_k)", min_value=1, max_value=100, value=20, step=1, help="How many of the top chunks go into the model prompt.")
run_clicked = col3.button("Ask", use_container_width=True)

clear_clicked = st.button("Clear", use_container_width=True)

if "history" not in st.session_state:
    st.session_state.history = []  # list of tuples: (role, query, params, response_json)

if clear_clicked:
    st.session_state.history = []
    st.rerun()


# ---------- Call API ----------
if run_clicked and query.strip():
    with st.spinner("Thinking…"):
        try:
            payload = {"query": query.strip(), "role": role, "top_k": int(top_k), "ctx_k": int(ctx_k)}
            resp = requests.post(API_URL, json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            st.session_state.history.append((role, query.strip(), dict(top_k=top_k, ctx_k=ctx_k), data))
        except requests.exceptions.RequestException as e:
            st.error(f"Request failed: {e}")
        except ValueError:
            st.error("Response was not valid JSON.")

# ---------- Render chat ----------
if not st.session_state.history:
    st.caption("Tip: increase *Top results (k)* if you want to see many chunks in Sources.")
else:
    for r, q, params, data in reversed(st.session_state.history):
        st.markdown(f"**Role:** `{r}`")
        st.markdown(f"**You:** {q}")
        st.markdown(f"**Answer:** {data.get('answer','(no answer)')}")

        sources = data.get("sources", [])
        with st.expander(f"Sources ({len(sources)})", expanded=True):
            for i, s in enumerate(sources, start=1):
                domain = s.get("domain", "unknown")
                title = (s.get("title") or os.path.basename(s.get("source") or "") or "Untitled").strip()
                path = s.get("source") or ""
                try:
                    rel = os.path.relpath(path, start=str(Path.cwd()))
                except Exception:
                    rel = path

                score = float(s.get("score", 0.0))
                chunk_id = s.get("chunk_id")
                chunk_label = f"chunk {chunk_id}" if chunk_id is not None else f"rank {i}"

                st.markdown(f"**[{i}]** `{domain}` — **{title}**  ·  _{chunk_label}_  ·  score={score:.4f}")
                if rel:
                    st.caption(rel)
                snippet = (s.get("snippet") or "").strip()
                if snippet:
                    # show full snippet (no grouping, no trimming other than a big upper bound)
                    st.write(snippet[:2000] + ("…" if len(snippet) > 2000 else ""))

# ---------- Footer ----------
st.caption(f"Backend: {API_URL}")

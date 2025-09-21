# FinRAG-MCP

**FinRAG-MCP** is a **Role-Based Access Control (RBAC) Retrieval-Augmented Generation (RAG)** system, integrated with the **Model Context Protocol (MCP)**.  
It lets you query company documents (Engineering, Finance, HR, Marketing, General) securely — each role only sees what it’s allowed to.  

✅ Works with **Claude Desktop (MCP tools)**  
✅ Optional **Streamlit UI** + **FastAPI gateway**  
✅ Uses **Qdrant (local)** for vector search  
✅ Documents enriched with **metadata + citations**

---

## 🚀 Why this project
- **Secure answers** → Employees, Managers, and C-Level see different data (RBAC).  
- **Trusted output** → Every response includes sources and file chunks.  
- **Flexible** → Claude MCP integration + standalone UI.  
- **Practical** → Handles Markdown, CSV, reports, handbooks, financial summaries.

---

## 📂 Project Structure

# FinRAG-MCP

**FinRAG-MCP** is a **Role-Based Access Control (RBAC) Retrieval-Augmented Generation (RAG)** system, integrated with the **Model Context Protocol (MCP)**.  
It lets you query company documents (Engineering, Finance, HR, Marketing, General) securely — each role only sees what it’s allowed to.  

✅ Works with **Claude Desktop (MCP tools)**  
✅ Optional **Streamlit UI** + **FastAPI gateway**  
✅ Uses **Qdrant (local)** for vector search  
✅ Documents enriched with **metadata + citations**

---

## 🚀 Why this project
- **Secure answers** → Employees, Managers, and C-Level see different data (RBAC).  
- **Trusted output** → Every response includes sources and file chunks.  
- **Flexible** → Claude MCP integration + standalone UI.  
- **Practical** → Handles Markdown, CSV, reports, handbooks, financial summaries.

---

## 📂 Project Structure



---

## ⚙️ Setup
``bash
# Clone repo
git clone https://github.com/YOUR_GITHUB/finrag-mcp.git
cd finrag-mcp

# Create venv with uv
uv venv .venv
source .venv/bin/activate
uv sync

# Add secrets
cat > env/.env << 'EOF'
OPENAI_API_KEY=sk-REPLACE_ME
OPENAI_MODEL=gpt-4o-mini
EMBED_MODEL=text-embedding-3-large
QDRANT_LOCAL_PATH=.qdrant_local
QDRANT_COLLECTION_PREFIX=finrag
FINRAG_ROLE=EMPLOYEE
EOF

# Ingest docs
uv run python -m ingest.run_ingest --data-root ./data



Run Options
1) Claude MCP (Recommended)
Open Claude Desktop → Settings → Developer → Local MCP servers
Add:
Command: .../finrag-mcp/.venv/bin/python
Args: -m mcp_server.server
Env: from .env
Claude can now call tools:
set_role("FINANCE")
search("Q4 2024 revenue drivers", top_k=5)

# Start gateway
uv run uvicorn gateway.app:app --port 8000

# Start UI
uv run streamlit run ui/app.py --server.port 8501


Example Queries
Employee: “What does the handbook say about leave approval?”
Engineering: “List services in the architecture doc.”
Finance: “Summarize Q4 revenue drivers.”
HR: “What’s the rule for sick leave >2 days?”
Marketing: “Which 2024 campaigns had the best ROI?”





# ingest/run_ingest.py
import os, argparse, uuid, csv, fnmatch
from pathlib import Path
from dotenv import load_dotenv
import yaml
from collections import defaultdict

from qdrant_client.http.models import PointStruct
from vectorstore.qdrant_init import connect_qdrant, ensure_collection
from ingest.loaders import load_file
from ingest.splitters import split_text
from ingest.embed import Embedder

DEFAULT_DOMAINS = ["engineering","finance","hr","marketing","general"]

FOLDER_ALLOW = {
    "engineering": ["ENGINEERING", "C_LEVEL"],
    "finance":     ["FINANCE", "C_LEVEL"],
    "hr":          ["HR", "C_LEVEL"],
    "marketing":   ["MARKETING", "C_LEVEL"],
    "general":     ["EMPLOYEE","ENGINEERING","FINANCE","HR","MARKETING","C_LEVEL"],
}

# ---------- metadata helpers ----------
def load_metadata(meta_dir: Path) -> dict:
    data = {"defaults": {"allow": ["EMPLOYEE"]}, "rules": []}
    yml = meta_dir / "metadata.yaml"
    if yml.exists():
        y = yaml.safe_load(yml.read_text(encoding="utf-8")) or {}
        data["defaults"] = y.get("defaults", data["defaults"])
        for r in (y.get("files") or []):
            data["rules"].append(r)
    csvp = meta_dir / "metadata.csv"
    if csvp.exists():
        with csvp.open() as f:
            rdr = csv.DictReader(f)
            for r in rdr:
                rule = {
                    "path": r.get("path") or None,
                    "glob": r.get("glob") or None,
                    "allow": [a.strip() for a in (r.get("allow") or "").split(";") if a.strip()] or None,
                    "tags":  [t.strip() for t in (r.get("tags")  or "").split(";") if t.strip()] or None,
                    "doc_type": r.get("doc_type") or None,
                    "title": r.get("title") or None,
                    "boost": float(r.get("boost") or 1.0),
                    "skip": (r.get("skip","").lower() == "true"),
                }
                data["rules"].append(rule)
    return data

def resolve_meta(meta: dict, rel_path_from_data: str, folder_domain: str) -> dict:
    out = {
        "allow": FOLDER_ALLOW.get(folder_domain, ["EMPLOYEE"]),
        "tags": [],
        "doc_type": None,
        "title": None,
        "boost": 1.0,
        "skip": False,
    }
    for r in meta.get("rules", []):
        hit = False
        if r.get("path") and r["path"] == rel_path_from_data:
            hit = True
        elif r.get("glob") and fnmatch.fnmatch(rel_path_from_data, r["glob"]):
            hit = True
        if not hit:
            continue
        if r.get("allow"):
            out["allow"] = r["allow"]
        for k in ["tags","doc_type","title","boost","skip"]:
            if r.get(k) not in (None, [], "", False):
                out[k] = r[k]
    return out

def iter_docs(root: Path):
    for path in root.rglob("*"):
        if path.is_file() and not path.name.startswith("."):
            yield path

# ---------- main ----------
def main():
    load_dotenv("env/.env")

    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--collections", nargs="*", default=DEFAULT_DOMAINS)
    args = ap.parse_args()

    data_root = Path(args.data_root).resolve()
    prefix = os.getenv("QDRANT_COLLECTION_PREFIX", "finrag")
    host   = os.getenv("QDRANT_HOST", "localhost")
    port   = int(os.getenv("QDRANT_PORT", "6333"))
    chunk  = int(os.getenv("CHUNK_SIZE", "1200"))
    overlap= int(os.getenv("CHUNK_OVERLAP", "150"))
    embed_model = os.getenv("EMBED_MODEL", "text-embedding-3-large")

    client = connect_qdrant(host, port)
    embedder = Embedder(model=embed_model)

    meta = load_metadata(data_root / "_metadata")

    # ensure all base collections exist
    for d in DEFAULT_DOMAINS:
        ensure_collection(client, f"{prefix}_{d}", dim=3072)

    for domain in args.collections:
        domain_dir = data_root / domain
        if not domain_dir.exists():
            print(f"[skip] {domain_dir} not found")
            continue

        print(f"\n[ingest] {domain} -> {prefix}_{domain}")

        # Bucket rows per target collection so we can upsert once per collection
        buckets: dict[str, list[tuple[str, dict]]] = defaultdict(list)

        for path in iter_docs(domain_dir):
            rel_from_data = str(path.relative_to(data_root))
            try:
                raw = load_file(path)
                chunks = split_text(raw, chunk, overlap)
                m = resolve_meta(meta, rel_from_data, domain)
                if m["skip"]:
                    continue

                # Build the base payload once
                for idx, chunk_text in enumerate(chunks):
                    payload = {
                        "source": str(path),
                        "domain": domain,
                        "chunk_id": idx,
                        "role_allow": m["allow"],
                        "text": chunk_text,
                        "tags": m.get("tags") or [],
                        "doc_type": m.get("doc_type"),
                        "title": m.get("title"),
                        "boost": float(m.get("boost", 1.0)),
                    }

                    # Always index in its own domain
                    buckets[f"{prefix}_{domain}"].append((chunk_text, payload))

                    # If metadata says EMPLOYEE can see it, also publish to GENERAL
                    if "EMPLOYEE" in payload["role_allow"] and domain != "general":
                        buckets[f"{prefix}_general"].append((chunk_text, payload))

            except Exception as e:
                print(f"[warn] failed to process {rel_from_data}: {e}")

        # Upsert per collection
        for collection, rows in buckets.items():
            texts  = [t for t,_ in rows]
            ploads = [p for _,p in rows]
            vectors = embedder.embed(texts)
            n = min(len(vectors), len(ploads))
            points = [PointStruct(id=str(uuid.uuid4()), vector=vectors[i], payload=ploads[i]) for i in range(n)]
            ensure_collection(client, collection, dim=3072)
            client.upsert(collection_name=collection, points=points)
            print(f"[ok] upserted {len(points)} chunks into {collection}")

if __name__ == "__main__":
    main()

from typing import List

def split_text(text: str, chunk_size: int = 1200, overlap: int = 150) -> List[str]:
    # simple paragraph-first, then length guard
    paras = [p.strip() for p in text.splitlines() if p.strip()]
    chunks, cur = [], ""
    for p in paras:
        if len(cur) + len(p) + 1 <= chunk_size:
            cur = (cur + "\n" + p) if cur else p
        else:
            if cur:
                chunks.append(cur)
            # start next chunk with overlap
            cur_tail = cur[-overlap:] if overlap and len(cur) > overlap else ""
            cur = (cur_tail + "\n" + p).strip()
    if cur:
        chunks.append(cur)
    return chunks

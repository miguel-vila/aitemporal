import re
import spacy

# python -m spacy download es_core_news_lg
nlp = spacy.load("es_core_news_lg")

SEP_RE = re.compile(r"\s*[-–—:|•]\s*")
EP_RE = re.compile(r"(?:^|\s)[\(#]?\s*(?:ep\.?|episodio)?\s*\d+\)?", re.IGNORECASE)

def normalize(title: str) -> str:
    t = re.sub(r"[–—]", "-", title)
    t = EP_RE.sub(" ", t)              # remove episode markers anywhere
    t = re.sub(r"\s{2,}", " ", t).strip()
    return t

def looks_like_name(s: str) -> bool:
    # Accept strings with 2–5 tokens, mostly Title Case (allow accents/ñ), allow de/del/la/da/di.
    toks = [w for w in re.split(r"\s+", s) if w]
    if not (1 <= len(toks) <= 6): 
        return False
    lowers_ok = {"de","del","la","las","los","y","da","di","do","dos"}
    good = 0
    for w in toks:
        if w.lower() in lowers_ok: 
            good += 1
        elif re.match(r"^[A-ZÁÉÍÓÚÑÜ][a-záéíóúñü]+\.?$", w):
            good += 1
    return good >= max(2, len(toks)-1)

def extract_names(title: str) -> list[str]:
    t = normalize(title)
    parts = [p.strip() for p in SEP_RE.split(t) if p.strip()]
    # heuristic pick: prefer first part that looks like a name; else first part
    cand = None
    for p in parts:
        if looks_like_name(p) and not any(ch in p for ch in "?!"):
            cand = p; break
    if cand is None:
        cand = parts[0]

    # NER validation on candidate and full title (in case name spans weirdly)
    doc_cand = nlp(cand)
    pers = [ent.text for ent in doc_cand.ents if ent.label_ == "PER"]
    if pers:
        return pers

    # fallback: run on whole title
    doc_all = nlp(t)
    pers_all = [ent.text for ent in doc_all.ents if ent.label_ == "PER"]
    if pers_all:
        return pers_all

    # final fallback: return candidate if it looks like a name
    return [cand] if looks_like_name(cand) else []

# Examples
titles = [
    "#128 - Alejandro Salazar - La ventaja colombiana, la tragedia, y la tecnocracia",
    "Alfonso Gómez Méndez - ¿Una guerra en vano? - #186",
]

# from transcript_db import TranscriptDB 
# import asyncio

# async def main():    
#     db = TranscriptDB()
#     vids = await db.get_unprocessed_videos()
#     print(f"Found {len(vids)} unprocessed videos")
#     for video in vids:
#         print(video.title, "->", extract_names(video.title))

# if __name__ == "__main__":
#    asyncio.run(main())

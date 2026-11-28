"""
Readability signals for Hindi, Tamil and Telugu.

Flesch-Kincaid is calibrated for English and must not be used on these languages. What is measured here is ADVISORY:
sentence length, length of words in aksharas (orthographic syllables) and the share of long words. The targets are
engineering guesses (agglutinative Tamil/Telugu words are longer and sentences have fewer words than Hindi), NOT
calibrated against comprehension data. The only valid evidence of "understandable" is a comprehension test with real
gig workers and a native-speaker review - this module just helps a reviewer find the hard sentences.
"""
import re

SCRIPT = {   # consonant ranges, independent-vowel ranges, virama, whether virama joins consonants into one akshara
    "hi": dict(cons=[(0x0915, 0x0939), (0x0958, 0x095F)], vow=[(0x0904, 0x0914), (0x0960, 0x0961)], virama=0x094D, conj=True),
    "ta": dict(cons=[(0x0B95, 0x0BB9)], vow=[(0x0B85, 0x0B94)], virama=0x0BCD, conj=False),
    "te": dict(cons=[(0x0C15, 0x0C39), (0x0C58, 0x0C5A)], vow=[(0x0C05, 0x0C14), (0x0C60, 0x0C61)], virama=0x0C4D, conj=True),
}
TARGETS = {"hi": dict(sentence_words=20, long_word_share=0.12), "ta": dict(sentence_words=14, long_word_share=0.30),
           "te": dict(sentence_words=14, long_word_share=0.30)}          # UNCALIBRATED
LONG_AKSHARAS = 7
URL = re.compile(r"(?:[\w-]+\.)+(?:gov\.in|nic\.in|org|in|com)\b")
INDIC_WORD = re.compile(r"[\u0900-\u097F\u0B80-\u0BFF\u0C00-\u0C7F]+")

def _in(cp, ranges): return any(lo <= cp <= hi for lo, hi in ranges)

def aksharas(word: str, lang: str) -> int:
    s, n, joined = SCRIPT[lang], 0, False
    for ch in word:
        cp = ord(ch)
        if _in(cp, s["cons"]):
            if not (joined and s["conj"]): n += 1
            joined = False
        elif _in(cp, s["vow"]): n += 1; joined = False
        elif cp == s["virama"]: joined = True
        else: joined = False
    return n

def readability_indic(text: str, lang: str) -> dict:
    protected = URL.sub("URL", text)
    sents = [x.strip() for x in re.split(r"[।.!?\n]+", protected) if INDIC_WORD.search(x)]
    words = [w for s in sents for w in INDIC_WORD.findall(s)]
    if not words: return {"words": 0, "sentences": 0}
    ak = [aksharas(w, lang) for w in words]
    t = TARGETS[lang]
    per_sent = [len(INDIC_WORD.findall(s)) for s in sents]
    long_sents = [s for s, k in zip(sents, per_sent) if k > t["sentence_words"]]
    out = {"words": len(words), "sentences": len(sents), "avg_sentence_words": round(len(words) / len(sents), 1),
           "avg_aksharas_per_word": round(sum(ak) / len(ak), 2), "long_word_share": round(sum(a >= LONG_AKSHARAS for a in ak) / len(ak), 2),
           "long_sentences": long_sents, "targets": t, "calibrated": False}
    out["within_targets"] = (out["avg_sentence_words"] <= t["sentence_words"] and out["long_word_share"] <= t["long_word_share"])
    return out

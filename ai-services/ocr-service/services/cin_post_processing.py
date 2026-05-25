"""
services/cin_post_processing.py
Post-traitement textuel des champs CIN (recto + verso).

Fonctions exportées :
  - normalize_dob(texts_ar, texts_en, scores_ar, scores_en) -> dict
  - postprocess_cin_field(label, texts, scores) -> tuple[list[str], list[float]]
"""
import re

# ── Mois arabes ───────────────────────────────────────────────────────────────
_AR_MONTHS: dict[str, str] = {
    "جانفي": "01", "فيفري": "02", "مارس": "03", "أفريل": "04",
    "افريل": "04", "ماي":   "05", "جوان":  "06", "جويلية": "07",
    "اوت":   "08", "أوت":   "08", "سبتمبر": "09", "اكتوبر": "10",
    "أكتوبر": "10", "نوفمبر": "11", "ديسمبر": "12",
}

# ── Regex ─────────────────────────────────────────────────────────────────────
_AR_DIACRITICS   = re.compile(r"[ً-ٰٟ]")
_AR_CHAR         = re.compile(r"[؀-ۿ]")
# Lettre arabe isolée en fin de token  ex: "طالب ا"  → "طالب"
_TRAILING_SINGLE = re.compile(r"\s+[؀-ۿ]$")
# Lettre arabe isolée en début de token  ex: "ا محمد"  → "محمد"
_LEADING_SINGLE  = re.compile(r"^[؀-ۿ]\s+")

# Tokens imprimés sur la CIN verso détectés par l'OCR à supprimer
_ADDRESS_LABEL_TOKENS: frozenset[str] = frozenset({"العنوان", "العنران", "العنوأن"})

# Champs arabes textuels CIN concernés par le nettoyage des lettres parasites
_TEXT_FIELDS_CIN: frozenset[str] = frozenset({
    "full_name", "address", "mother_name", "profession",
    "nom_complet", "nom_mere",
})


# ── Helpers internes ──────────────────────────────────────────────────────────

def _strip_diacritics(text: str) -> str:
    return _AR_DIACRITICS.sub("", text).strip()


def _find_month(text: str) -> str:
    t = _strip_diacritics(text)
    if t in _AR_MONTHS:
        return t
    for month in _AR_MONTHS:
        if month in t or t in month:
            return month
    return ""


def _clean_token(token: str) -> str:
    """Supprime les lettres arabes isolées collées en début ou fin d'un token."""
    token = _TRAILING_SINGLE.sub("", token)
    token = _LEADING_SINGLE.sub("", token)
    return token.strip()


# ── API publique ──────────────────────────────────────────────────────────────

def normalize_dob(
    texts_ar: list[str],
    texts_en: list[str],
    scores_ar: list[float],
    scores_en: list[float],
) -> dict:
    """Normalise une date CIN (mois arabe + chiffres) vers 'JJ مois AAAA'."""
    import numpy as np

    month_ar = ""
    for t in texts_ar:
        found = _find_month(t)
        if found:
            month_ar = found
            break
    if not month_ar:
        for t in texts_en:
            found = _find_month(t)
            if found:
                month_ar = found
                break

    raw = "".join(t.strip() for t in texts_en)
    numbers = re.findall(r"\d+", raw)

    if len(numbers) == 1:
        n = numbers[0]
        if len(n) == 5:
            head4_ok = 1900 < int(n[:4]) < 2100
            tail4_ok = 1900 < int(n[1:]) < 2100
            if head4_ok and not tail4_ok:
                # ex: "20265" → année=2026, jour=5
                numbers = [n[4:], n[:4]]
            elif tail4_ok and not head4_ok:
                # ex: "52026" → jour=5, année=2026
                numbers = [n[:1], n[1:]]
            elif head4_ok and tail4_ok:
                # ambiguïté : préfère l'année en tête (format AAAA+J)
                numbers = [n[4:], n[:4]]
            else:
                # Cherche un sous-groupe de 4 chiffres consécutifs qui forment une année valide
                for start in range(len(n) - 3):
                    candidate = int(n[start:start + 4])
                    if 1900 < candidate < 2100:
                        residual = (n[:start] + n[start + 4:]).lstrip("0") or "0"
                        numbers = [residual, str(candidate)]
                        break
        elif len(n) == 6:
            year4_head = int(n[:4])
            year4_tail = int(n[2:])
            if 1900 < year4_head < 2100 and not (1900 < year4_tail < 2100):
                numbers = [n[4:], n[:4]]
            elif 1900 < year4_tail < 2100:
                numbers = [n[:2], n[2:]]
            elif 1900 < year4_head < 2100:
                numbers = [n[4:], n[:4]]
            else:
                numbers = [n[:2], n[2:]]
        elif len(n) == 8:
            if int(n[:4]) > 1900:
                numbers = [n[6:], n[:4]]
            else:
                numbers = [n[:2], n[4:]]

    if len(numbers) > 1:
        year_from_long = ""
        standalone = []
        for n in numbers:
            if len(n) >= 5:
                if int(n[:4]) > 1900:
                    year_from_long = n[:4]
                elif int(n[-4:]) > 1900:
                    year_from_long = n[-4:]
            else:
                standalone.append(n)
        if year_from_long:
            numbers = standalone + [year_from_long]

    year, day = "", ""
    for n in numbers:
        if len(n) == 4 and int(n) > 1900:
            year = n
        elif len(n) <= 2 and not day:
            day = n
    if not year and len(numbers) >= 2:
        if len(numbers[-1]) == 4 and int(numbers[-1]) > 1900:
            year = numbers[-1]
            day  = numbers[0]
        else:
            day = numbers[0]
    elif not year and len(numbers) == 1:
        day = numbers[0]

    parts = []
    if day and month_ar and year:
        parts = [day, month_ar, year]
    elif day and year:
        parts = [day, year]
    elif day and month_ar:
        parts = [day, month_ar]
    elif month_ar:
        parts = [month_ar]
    elif day:
        parts = [day]

    parts = [p for p in parts if p]
    if not parts:
        return {"ocr_text": None, "ocr_score": None, "ocr_lines": None}

    all_scores = scores_ar + scores_en
    avg_score = round(float(np.mean(all_scores)) if all_scores else 0.8, 4)
    return {
        "ocr_text":  " ".join(parts),
        "ocr_score": avg_score,
        "ocr_lines": [{"text": p, "score": avg_score} for p in parts],
    }


def postprocess_cin_field(
    label: str,
    texts: list[str],
    scores: list[float],
) -> tuple[list[str], list[float]]:
    """
    Applique tous les post-traitements textuels CIN pour un champ donné.

    Traitements :
      1. Suppression du token imprimé العنوان (champ address)
      2. Nettoyage des lettres arabes isolées collées en début/fin de token
         (ex: "طالب ا" → "طالب")
    """
    # 1. Suppression du label العنوان pour le champ address
    if label == "address":
        filtered = [
            (t, s) for t, s in zip(texts, scores)
            if t.strip() not in _ADDRESS_LABEL_TOKENS
        ]
        if filtered:
            texts  = [f[0] for f in filtered]
            scores = [f[1] for f in filtered]

    # 2. Nettoyage lettres parasites pour tous les champs textuels arabes
    if label in _TEXT_FIELDS_CIN:
        cleaned = []
        kept_scores = []
        for t, s in zip(texts, scores):
            c = _clean_token(t)
            if c:
                cleaned.append(c)
                kept_scores.append(s)
        if cleaned:
            texts  = cleaned
            scores = kept_scores

    return texts, scores

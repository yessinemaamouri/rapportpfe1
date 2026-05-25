"""
services/post_ocr_passport.py
Post-traitement OCR passeport tunisien.

1. Extraction nom_ar / prenom_ar depuis full_name_ar
   Format attendu :
     Homme         : <prenom> بن  <nom_pere [composé]> <laqab>
     Femme         : <prenom> بنت <nom_pere [composé]> <laqab>
     Femme mariée  : <prenom> بنت <nom_pere [composé]> <laqab> حرم <...>

2. Normalisation des dates (dob, issue_date, expiry_date) → JJ/MM/AAAA
   - Chiffres arabes (٠-٩) → latins
   - Séparateurs variés (- . espace) → /
   - Mois en texte fr/ar → numéro à 2 chiffres
   - Année 2 chiffres → 4 chiffres (>= 30 → 19xx, < 30 → 20xx)
   - Format ISO AAAA/MM/JJ → JJ/MM/AAAA
   - OCR collé 8 chiffres JJMMAAAA → JJ/MM/AAAA
"""
from __future__ import annotations

import re
from typing import Optional

# ── Noms arabes ───────────────────────────────────────────────────────────────

_SEP_HARAM = "حرم"
_SEP_BENT  = "بنت"
_SEP_BEN   = "بن"

# ── Dates ─────────────────────────────────────────────────────────────────────

_AR_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

_MONTH_TEXT: dict[str, str] = {
    # Français
    "janvier": "01", "fevrier": "02", "février": "02", "mars": "03",
    "avril": "04", "mai": "05", "juin": "06", "juillet": "07",
    "aout": "08", "août": "08", "septembre": "09", "octobre": "10",
    "novembre": "11", "decembre": "12", "décembre": "12",
    # Abréviations fr
    "jan": "01", "fev": "02", "fév": "02", "avr": "04",
    "jun": "06", "jul": "07", "aou": "08", "sep": "09",
    "oct": "10", "nov": "11", "dec": "12", "déc": "12",
    # Arabe tunisien
    "جانفي": "01", "فيفري": "02", "مارس": "03", "افريل": "04", "أفريل": "04",
    "ماي": "05", "جوان": "06", "جويلية": "07", "أوت": "08", "اوت": "08",
    "سبتمبر": "09", "أكتوبر": "10", "اكتوبر": "10", "نوفمبر": "11", "ديسمبر": "12",
}

_DATE_FIELDS = ("dob", "issue_date", "expiry_date")


def _expand_year(yy: int) -> str:
    """Année 2 chiffres → 4 chiffres. Pivot 30 : >= 30 → 19xx, < 30 → 20xx."""
    return f"19{yy:02d}" if yy >= 30 else f"20{yy:02d}"


def normalize_date(raw: Optional[str]) -> Optional[str]:
    """
    Normalise une date OCR brute vers le format JJ/MM/AAAA.
    Retourne None si non reconnu.
    """
    if not raw or not raw.strip():
        return None

    s = raw.strip().translate(_AR_DIGITS)  # chiffres arabes → latins

    # ── Cas : mois en texte  "23 juin 1992" / "23 جوان 1992" ─────────────────
    m = re.match(r"^(\d{1,2})[\s\-/.](\S+)[\s\-/.](\d{2,4})$", s)
    if m:
        day, month_raw, year_raw = m.group(1), m.group(2), m.group(3)
        month = _MONTH_TEXT.get(month_raw.lower()) or _MONTH_TEXT.get(month_raw)
        if month:
            yy = int(year_raw)
            year = year_raw if len(year_raw) == 4 else _expand_year(yy)
            return f"{int(day):02d}/{month}/{year}"

    # ── Remplace tous les séparateurs par / ──────────────────────────────────
    s = re.sub(r"[\-.\s]", "/", s)
    s = re.sub(r"/+", "/", s).strip("/")

    parts = s.split("/")

    # ── 3 parties ─────────────────────────────────────────────────────────────
    if len(parts) == 3:
        a, b, c = parts[0], parts[1], parts[2]

        # AAAA/MM/JJ → JJ/MM/AAAA
        if len(a) == 4 and a.isdigit():
            day, month, year = c, b, a
        else:
            day, month, year = a, b, c

        # Année 2 chiffres → 4 chiffres
        if len(year) == 2 and year.isdigit():
            year = _expand_year(int(year))

        try:
            return f"{int(day):02d}/{int(month):02d}/{year}"
        except ValueError:
            return None

    # ── OCR collé : 8 chiffres JJMMAAAA ──────────────────────────────────────
    if len(parts) == 1 and re.match(r"^\d{8}$", parts[0]):
        d = parts[0]
        return f"{d[0:2]}/{d[2:4]}/{d[4:8]}"

    # ── OCR collé : 6 chiffres JJMMAA ────────────────────────────────────────
    if len(parts) == 1 and re.match(r"^\d{6}$", parts[0]):
        d = parts[0]
        year = _expand_year(int(d[4:6]))
        return f"{d[0:2]}/{d[2:4]}/{year}"

    return None


# ── Extraction noms arabes ────────────────────────────────────────────────────

def extract_arabic_names(full_name_ar: Optional[str]) -> dict:
    """Retourne {"nom_ar": str | None, "prenom_ar": str | None}."""
    if not full_name_ar or not full_name_ar.strip():
        return {"nom_ar": None, "prenom_ar": None}

    name = full_name_ar.strip()

    # Règle 1 — femme mariée : ignore tout à partir de "حرم"
    if _SEP_HARAM in name:
        name = name[:name.index(_SEP_HARAM)].strip()

    # Règle 2 — split sur le premier séparateur de filiation
    sep_found = None
    for sep in (_SEP_BENT, _SEP_BEN):
        if sep in name:
            sep_found = sep
            break

    if sep_found is None:
        tokens = name.split()
        if len(tokens) == 1:
            return {"nom_ar": None, "prenom_ar": tokens[0]}
        return {"nom_ar": tokens[-1], "prenom_ar": tokens[0]}

    left, right = name.split(sep_found, maxsplit=1)
    prenom_ar    = left.strip() or None
    right_tokens = right.strip().split()
    nom_ar       = right_tokens[-1] if right_tokens else None

    return {"nom_ar": nom_ar, "prenom_ar": prenom_ar}


# ── Post-traitement OCR brut passeport (fragments PaddleOCR) ─────────────────

# Champs arabes passeport avec disposition multi-fragments RTL
_RTL_MULTILINE_PASSPORT: frozenset[str] = frozenset({"full_name", "address"})


def normalize_latin_date(texts: list[str], scores: list[float]) -> dict:
    """Normalise une date passeport en chiffres latins purs (PaddleOCR) vers 'JJ MM AAAA'."""
    import numpy as np

    raw    = "".join(t.strip() for t in texts)
    digits = re.sub(r"\D", "", raw)

    day = month = year = ""
    if len(digits) == 8:
        day, month, year = digits[:2], digits[2:4], digits[4:]
    elif len(digits) == 6:
        day, month, year = digits[:2], digits[2:4], digits[4:]
    else:
        nums = re.findall(r"\d+", raw)
        if len(nums) >= 3:
            day, month, year = nums[0], nums[1], nums[2]
        elif len(nums) == 2:
            day, month = nums[0], nums[1]
        elif len(nums) == 1:
            day = nums[0]

    parts = [p for p in (day, month, year) if p]
    if not parts:
        return {"ocr_text": None, "ocr_score": None, "ocr_lines": None}

    avg_score = round(float(np.mean(scores)) if scores else 0.0, 4)
    return {
        "ocr_text":  " ".join(parts),
        "ocr_score": avg_score,
        "ocr_lines": [{"text": p, "score": avg_score} for p in parts],
    }


def postprocess_passport_rtl(
    label: str,
    texts: list[str],
    scores: list[float],
    boxes: list,
) -> tuple[list[str], list[float]]:
    """
    Tri RTL multi-lignes pour les champs arabes passeport (full_name, address).
    Regroupe les fragments par ligne (y_min ±20px) puis trie de droite à gauche.
    """
    if label not in _RTL_MULTILINE_PASSPORT:
        return texts, scores
    if not texts or len(texts) <= 1 or not boxes:
        return texts, scores

    combined = list(zip(texts, scores, boxes))
    combined.sort(key=lambda x: x[2][1])  # tri par y_min (haut → bas)

    line_height = 20
    lines_grouped: list[list] = []
    current_line = [combined[0]]
    for item in combined[1:]:
        if abs(item[2][1] - current_line[0][2][1]) <= line_height:
            current_line.append(item)
        else:
            lines_grouped.append(sorted(current_line, key=lambda x: x[2][0], reverse=True))
            current_line = [item]
    lines_grouped.append(sorted(current_line, key=lambda x: x[2][0], reverse=True))

    combined = [item for line in lines_grouped for item in line]
    return [c[0] for c in combined], [c[1] for c in combined]


# ── Point d'entrée principal ──────────────────────────────────────────────────

def enrich_passport_structured(structured: dict) -> dict:
    """
    Enrichit le dict structured OCR passeport :
    - nom_ar / prenom_ar depuis full_name_ar
    - dates normalisées JJ/MM/AAAA pour dob, issue_date, expiry_date
    Modifie le dict en place et le retourne.
    """
    # Noms arabes
    names = extract_arabic_names(structured.get("full_name_ar"))
    structured["nom_ar"]    = names["nom_ar"]
    structured["prenom_ar"] = names["prenom_ar"]

    # Dates
    for field in _DATE_FIELDS:
        raw = structured.get(field)
        if raw:
            normalized = normalize_date(raw)
            if normalized:
                structured[field] = normalized

    return structured

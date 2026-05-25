"""
services/utils/report.py
Génération du rapport visuel final KYC (PNG).
Conserve exactement le même rendu que la fonction create_final_report() originale.
"""
import cv2
import numpy as np
import arabic_reshaper
from bidi.algorithm import get_display
from PIL import Image, ImageDraw, ImageFont
from loguru import logger

# Ordre d'affichage des champs par cin_type
REPORT_FIELD_ORDER: dict[str, list[str]] = {
    "recto": [
        "header_text", "num_cin", "last_name", "first_name",
        "full_name", "dob", "pob",
    ],
    "verso": [
        "address", "mother_name", "profession", "issue_date", "print_id",
    ],
}

# Police arabe pour le rapport (doit être disponible dans le répertoire d'exécution)
FONT_PATH: str = "Amiri-Regular.ttf"


def fix_arabic_display(text: str) -> str:
    """
    Reshape et réordonne le texte arabe pour l'affichage visuel RTL (console/PIL).
    """
    if not text:
        return ""
    reshaped = arabic_reshaper.reshape(text)
    return get_display(reshaped)


def create_final_report(
    img_with_boxes: np.ndarray,
    results: list[dict],
    output_path: str = "RAPPORT_FINAL_PFE.png",
    cin_type: str = "recto",
) -> None:
    """
    Génère le rapport visuel final KYC au format PNG.

    Args:
        img_with_boxes: Image BGR annotée (np.ndarray) avec les bounding boxes YOLO.
        results:        Liste de dicts {"class": str, "text": str, "score": float}.
        output_path:    Chemin de sortie du PNG généré.
        cin_type:       "recto" ou "verso".
    """
    if img_with_boxes is None:
        logger.warning("create_final_report : img_with_boxes est None, rapport non généré.")
        return

    # Conversion BGR → PIL, redimensionnement à hauteur fixe 600px
    img_rgb = cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    ratio   = 600 / img_pil.size[1]
    img_pil = img_pil.resize(
        (int(img_pil.size[0] * ratio), 600),
        Image.Resampling.LANCZOS,
    )

    # Canvas blanc
    canvas_w = img_pil.size[0] + 900
    canvas_h = 900
    report   = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    report.paste(img_pil, (30, 150))
    draw = ImageDraw.Draw(report)

    # Chargement des polices (fallback sur la police par défaut PIL si absente)
    try:
        font_header = ImageFont.truetype(FONT_PATH, 40)
        font_val    = ImageFont.truetype(FONT_PATH, 34)
        font_lbl    = ImageFont.truetype(FONT_PATH, 22)
        font_score  = ImageFont.truetype(FONT_PATH, 24)
    except Exception:
        logger.warning(f"Police '{FONT_PATH}' introuvable, utilisation de la police par défaut.")
        font_header = font_val = font_lbl = font_score = ImageFont.load_default()

    # Configuration arabic_reshaper (supprime les diacritiques, active les ligatures)
    reshaper_config = arabic_reshaper.ArabicReshaper(
        configuration={"delete_harakat": True, "support_ligatures": True}
    )

    # En-tête du rapport
    x_start = img_pil.size[0] + 80
    draw.text((x_start, 60), "ANALYSE DES DONNÉES EXTRAITES",
              fill=(20, 40, 60), font=font_header)
    draw.line((x_start, 115, canvas_w - 50, 115), fill=(52, 152, 219), width=3)

    # Mapping class → résultat pour accès O(1)
    data_map: dict[str, dict] = {res["class"]: res for res in results}

    field_order = REPORT_FIELD_ORDER.get(cin_type, REPORT_FIELD_ORDER["recto"])

    y_offset = 180
    for key in field_order:
        res       = data_map.get(key)
        val_raw   = res["text"]  if res else "---"
        score_val = res["score"] if res else 0.0

        # Supprime les chiffres arabes orientaux (Eastern Arabic numerals)
        # pour n'afficher que les chiffres latins
        for arabic_digit in "٠١٢٣٤٥٦٧٨٩":
            val_raw = val_raw.replace(arabic_digit, "")

        display_text = reshaper_config.reshape(val_raw)

        # Label (avec ombre légère simulée par double rendu décalé)
        label_txt = f"{key.upper()} :"
        for offset in range(2):
            draw.text((x_start + offset, y_offset), label_txt,
                      fill=(100, 100, 100), font=font_lbl)

        # Valeur
        val_pos = (x_start + 230, y_offset - 10)
        for dx in range(2):
            draw.text((val_pos[0] + dx, val_pos[1]), display_text,
                      fill=(0, 0, 0), font=font_val)

        # Score (% en vert)
        if res:
            score_txt = f"{score_val * 100:.1f}%"
            for offset in range(3):
                draw.text((canvas_w - 120 + offset, y_offset + 5), score_txt,
                          fill=(0, 150, 0), font=font_score)

        # Séparateur horizontal
        draw.line((x_start, y_offset + 65, canvas_w - 50, y_offset + 65),
                  fill=(240, 240, 240), width=1)
        y_offset += 100

    report.save(output_path)
    logger.info(f"Rapport final généré : {output_path}")

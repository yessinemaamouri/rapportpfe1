# Évaluation du Service OCR — Partie Rapport PFE

## 1. Méthodologie d'évaluation

### 1.1 Protocole expérimental

L'évaluation du service OCR a été conduite sur un corpus de documents d'identité tunisiens réels, répartis en trois catégories :

| Document | Nombre d'images | Champs évalués |
|---|---|---|
| CIN Recto | 60 images | num_cin, last_name, first_name, full_name, dob, pob |
| CIN Verso | 60 images | mother_name, profession, address, issue_date, print_id |
| Passeport | 60 images | num_pass, num_cin, last_name, first_name, full_name, dob, pob, issue_date, expiry_date, profession, address |

Le choix de 60 images par document assure une base de comparaison équitable entre les trois types de documents.

Le pipeline évalué se décompose en deux étapes distinctes testées séparément :

- **OCR brut** (`pure_ocr`) : texte retourné directement par PaddleOCR PP-OCRv5 sans aucun traitement
- **OCR post-traité** (`structured`) : texte après application du module de post-traitement (normalisation des dates, tri RTL des fragments arabes, suppression des tokens parasites)

La vérité terrain (`ground_truth`) a été construite manuellement : le service OCR a d'abord généré un template pré-rempli sur les 60 images, puis chaque champ a été vérifié et corrigé image par image.

### 1.2 Pré-traitement des textes avant comparaison

Avant tout calcul de métrique, les textes prédits et les références sont normalisés selon les règles suivantes :

1. **Suppression des diacritiques arabes** (tashkeel) — absents sur les documents officiels tunisiens
2. **Normalisation du alef** : `أ`, `إ`, `آ` → `ا`
3. **Réduction des espaces multiples** en espace simple
4. **Conversion en minuscules** pour les caractères latins

Cette normalisation évite de pénaliser des variations orthographiques mineures sans impact sur la lisibilité.

---

## 2. Métriques utilisées

### 2.1 Taux de détection (Detection Rate — DR)

**Définition** : proportion d'images où le champ a été localisé et extrait par le modèle YOLO, indépendamment de la qualité du texte reconnu.

$$DR = \frac{\text{Nombre d'images où le champ est détecté (non null)}}{N_{total}}$$

Cette métrique évalue exclusivement la performance du modèle de détection YOLO, pas celle de l'OCR. Un DR inférieur à 0,90 indique que la zone du champ est fréquemment manquée lors de la détection.

---

### 2.2 Taux de correspondance exacte (Exact Match Rate — EMR)

**Définition** : proportion d'images où le texte reconnu est strictement identique à la référence, après normalisation.

$$EMR = \frac{\text{Nombre d'images où } \text{normalize}(\hat{y}) = \text{normalize}(y)}{N_{détectés}}$$

L'EMR est calculé séparément pour l'OCR brut et l'OCR post-traité. Le gain apporté par le post-traitement est mesuré par :

$$\Delta EMR = EMR_{structured} - EMR_{pure\_ocr}$$

Un ΔEMR positif confirme l'apport du module de post-traitement. Un ΔEMR négatif indiquerait une dégradation due au post-traitement.

---

### 2.3 Taux d'erreur caractère (Character Error Rate — CER)

**Définition** : mesure la distance entre le texte prédit et la référence au niveau du caractère, basée sur la distance de Levenshtein.

$$CER = \frac{S + D + I}{N_{chars}(y)}$$

| Symbole | Signification |
|---|---|
| S | Substitutions : caractère remplacé |
| D | Deletions : caractère supprimé dans la prédiction |
| I | Insertions : caractère ajouté dans la prédiction |
| N | Longueur en caractères de la référence |

Deux indicateurs sont rapportés : la **moyenne** (CER moyen sur l'ensemble des images) et le **90ème percentile** (p90, représentant les 10% de cas les plus difficiles).

Le CER est particulièrement adapté à l'arabe où une seule lettre mal reconnue peut changer complètement le sens d'un mot.

---

### 2.4 Taux d'erreur mot (Word Error Rate — WER)

**Définition** : même principe que le CER mais appliqué au niveau du mot.

$$WER = \frac{S_w + D_w + I_w}{N_{mots}(y)}$$

Le WER est particulièrement utile pour les champs multi-mots (`full_name`, `address`, `mother_name`, `pob`) où la perte ou l'ajout d'un mot entier est plus significatif qu'une simple erreur de caractère.

---

### 2.5 Calibration du score de confiance

**Définition** : mesure la fiabilité du score de confiance retourné par PaddleOCR comme indicateur d'erreur.

$$Calibration = \frac{1}{N} \sum_{i=1}^{N} \left| score_i - \mathbf{1}[\hat{y}_i = y_i] \right|$$

Une calibration proche de 0 signifie que le score prédit correctement les erreurs (score élevé → reconnaissance correcte). Une valeur élevée indique que le modèle est sur-confiant même lorsqu'il se trompe.

---

## 3. Résultats

### 3.1 CIN Recto

**Vue globale (60 images) :**

| Métrique | OCR Brut | OCR Post-traité | Gain |
|---|---|---|---|
| EMR moyen | 79,41% | **94,13%** | +14,72 pts |
| CER moyen | 8,41% | **1,17%** | −7,24 pts |

**Résultats par champ :**

| Champ | DR | EMR Brut | EMR Post-traité | ΔEMR | CER Post-traité | Score OCR moyen |
|---|---|---|---|---|---|---|
| num_cin | 100% | 100,0% | **100,0%** | +0,0 | 0,000 | 0,999 |
| last_name | 100% | 96,7% | **96,7%** | +0,0 | 0,004 | 0,960 |
| first_name | 95% | 98,3% | **98,3%** | +0,0 | 0,004 | 0,944 |
| full_name | 100% | 83,3% | **83,3%** | +0,0 | 0,018 | 0,939 |
| dob | 100% | 0,0% | **88,3%** | **+88,3** | 0,041 | 0,847 |
| pob | 93% | 98,2% | **98,2%** | +0,0 | 0,004 | 0,962 |

**Analyse :**

Le champ `dob` (date de naissance) illustre parfaitement l'apport du post-traitement : l'OCR brut retourne les chiffres et le nom du mois arabes collés sans séparateur (ex : `29افريل2004`), ce qui donne un EMR de 0% en comparaison directe. Le module de normalisation des dates reconstruit la forme attendue (`29 أفريل 2004`), portant l'EMR à 88,3%.

Les champs `num_cin`, `last_name`, `first_name` et `pob` atteignent des performances quasi-parfaites dès l'OCR brut, le post-traitement n'apportant pas de modification sur ces champs simples.

Le champ `full_name` reste à 83,3% en raison de la complexité de la structure arabe (prénom + particule de filiation بن/بنت + nom composé) qui peut générer des variantes d'ordre entre les fragments RTL.

Le taux de détection de `pob` (93%) indique que YOLO manque occasionnellement cette zone, généralement petite et située en bas de la carte.

---

### 3.2 CIN Verso

**Vue globale (60 images) :**

| Métrique | OCR Brut | OCR Post-traité | Gain |
|---|---|---|---|
| EMR moyen | 59,67% | **98,67%** | +39,00 pts |
| CER moyen | 17,25% | **0,25%** | −17,00 pts |

**Résultats par champ :**

| Champ | DR | EMR Brut | EMR Post-traité | ΔEMR | CER Post-traité | Score OCR moyen |
|---|---|---|---|---|---|---|
| mother_name | 100% | 98,3% | **100,0%** | +1,7 | 0,000 | 0,971 |
| profession | 100% | 100,0% | **100,0%** | +0,0 | 0,000 | 0,963 |
| address | 100% | 0,0% | **95,0%** | **+95,0** | 0,007 | 0,933 |
| issue_date | 100% | 0,0% | **98,3%** | **+98,3** | 0,005 | 0,853 |
| print_id | 100% | 100,0% | **100,0%** | +0,0 | 0,000 | 0,999 |

**Analyse :**

La CIN Verso présente les gains les plus spectaculaires. Le champ `address` passe de 0% à 95% grâce à deux traitements combinés : le tri RTL des fragments (PaddleOCR détecte les mots de gauche à droite alors que l'arabe s'écrit de droite à gauche) et la suppression du token imprimé `العنوان` (label physique sur la carte capté par l'OCR). Le champ `issue_date` suit la même logique que les dates du recto.

Les champs `profession` et `print_id` atteignent 100% dès l'OCR brut, confirmant que les champs courts et typographiquement distincts ne nécessitent pas de post-traitement.

---

### 3.3 Passeport

**Vue globale (60 images) :**

| Métrique | OCR Brut | OCR Post-traité | Gain |
|---|---|---|---|
| EMR moyen | 61,78% | **93,75%** | +31,97 pts |
| CER moyen | 14,77% | **2,37%** | −12,40 pts |

**Résultats par champ :**

| Champ | DR | EMR Brut | EMR Post-traité | ΔEMR | CER Post-traité | Score OCR moyen |
|---|---|---|---|---|---|---|
| num_pass | 100% | 100,0% | **100,0%** | +0,0 | 0,000 | 1,000 |
| num_cin | 100% | 100,0% | **100,0%** | +0,0 | 0,000 | 0,999 |
| last_name | 100% | 98,3% | **98,3%** | +0,0 | 0,011 | 0,996 |
| first_name | 100% | 98,3% | **98,3%** | +0,0 | 0,003 | 0,990 |
| full_name | 100% | 33,3% | **80,0%** | **+46,7** | 0,046 | 0,936 |
| dob | 100% | 0,0% | **100,0%** | **+100,0** | 0,000 | 0,971 |
| pob | 100% | 83,3% | **83,3%** | +0,0 | 0,074 | 0,947 |
| issue_date | 100% | 0,0% | **100,0%** | **+100,0** | 0,000 | 0,997 |
| expiry_date | 100% | 0,0% | **100,0%** | **+100,0** | 0,000 | 1,000 |
| profession | 98% | 72,9% | **78,0%** | +5,1 | 0,091 | 0,932 |
| address | 100% | 93,3% | **93,3%** | +0,0 | 0,036 | 0,966 |

**Analyse :**

Les trois champs de dates (`dob`, `issue_date`, `expiry_date`) atteignent 100% après post-traitement, partant de 0% en brut — confirmant que la normalisation des dates est une brique indispensable du pipeline.

Le champ `full_name` présente le gain le plus important (+46,7 pts) grâce au tri RTL qui réordonne les fragments arabes détectés dans le mauvais sens. Les 20% d'erreurs résiduelles correspondent principalement à des noms composés de 4 tokens ou plus où l'ambiguïté d'ordre persiste.

Le champ `profession` reste le plus difficile (EMR 78%) en raison du fond texturé du passeport qui perturbe la binarisation adaptative. Le CER de 9,1% indique que les erreurs sont partielles (quelques caractères), pas une incompréhension totale du champ.

Le champ `pob` (lieu de naissance) affiche un CER de 7,4% avec un p90 de 0,333 — les 10% de cas difficiles correspondent aux villes composées (`تونس العاصمة`, `بن عروس`) où un mot est parfois manqué.

---

## 4. Synthèse comparative

| Document | DR moyen | EMR Brut | EMR Post-traité | Gain ΔEMR | CER Post-traité |
|---|---|---|---|---|---|
| CIN Recto | **98,1%** | 79,4% | **94,1%** | +14,7 pts | **1,17%** |
| CIN Verso | **100,0%** | 59,7% | **98,7%** | +39,0 pts | **0,25%** |
| Passeport | **99,7%** | 61,8% | **93,8%** | +31,9 pts | **2,37%** |

**Observations transversales :**

1. **La détection YOLO est robuste** : DR > 98% sur tous les documents, avec seulement le champ `pob` du recto à 93% comme point d'amélioration identifié.

2. **Le post-traitement est indispensable** : sans lui, l'EMR moyen ne dépasse pas 62-79%. Il est particulièrement critique pour les champs dates (gain de 88 à 100 points) et les champs multi-fragments RTL (address, full_name).

3. **Les champs numériques sont parfaits** : `num_cin`, `num_pass`, `print_id` atteignent 100% EMR et CER=0 sur les trois documents — PaddleOCR reconnaît les chiffres avec une fiabilité maximale.

4. **Le CER post-traitement est excellent** : inférieur à 2,5% sur tous les documents, ce qui signifie qu'en moyenne moins d'un caractère sur 40 est erroné après post-traitement.

5. **La calibration des scores** est très bonne sur les champs simples (calibration < 0,05) mais dégradée sur les champs dates en mode brut (calibration ~0,97) — le modèle est très confiant même quand il retourne un format incorrect. Après post-traitement, la calibration des dates revient à des valeurs normales (< 0,03).

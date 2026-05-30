# 4.5 Déploiement et Intégration du Pipeline IA

## 4.5.1 Architecture générale des micro-services IA

Le pipeline IA de la plateforme eKYC constitue un **système IA unifié composé de 7 modules spécialisés et indépendants**, chacun dédié à une étape précise du parcours de vérification. Cette décomposition modulaire répond à trois objectifs : la **rapidité** (chaque module est optimisé pour une seule tâche et peut être exécuté sur des ressources dédiées), la **spécificité** (chaque modèle IA est entraîné et configuré pour un problème précis sans interférence avec les autres) et la **maintenabilité** (un module peut être mis à jour, remplacé ou réentraîné indépendamment sans impacter le reste du système).

### Vue d'ensemble des services

| Service | Port | Rôle | Réseau | Statut |
|---|---|---|---|---|
| `capture-cin` | 8001 | Capture temps réel CIN (recto + verso) via webcam | Public + AI-interne | Actif |
| `classification-service` | 8002 | Classification du type de document | AI-interne | Actif |
| `ocr-service` | 8003 | Extraction OCR CIN et passeport (YOLO + PaddleOCR) | AI-interne | Actif |
| `face-match` | 8005 | Comparaison biométrique selfie/document | AI-interne | Actif |
| `capture-passport` | 8006 | Capture temps réel passeport via webcam | Public + AI-interne | Actif |
| `capture-face` | 8007 | Capture selfie avec détection visage | Public + AI-interne | Actif |
| `ocr-credit` | 8004 | Extraction OCR attestations de travail / revenus (PaddleOCR sans YOLO) | AI-interne | Actif |

### Communication entre services

Tous les services tournent sur la même machine et communiquent via **HTTP sur localhost**, chacun écoutant sur un port dédié. Le frontend React appelle directement les services de capture et d'OCR ; le backend n'intervient que pour la soumission finale du dossier KYC et pour le filtrage biométrique inter-clients (appel face-match lors de la validation admin).

```
Frontend React
      │
      │ WebSocket / HTTP direct
      ▼
┌─────────────────────────────────────────────────────┐
│                 localhost                           │
│                                                     │
│  Backend API          :8000  ← Frontend (submit)    │
│  capture-cin          :8001  ← Frontend direct      │
│  classification       :8002  ← Frontend direct      │
│  ocr-service          :8003  ← Frontend direct      │
│  ocr-credit           :8004  ← Frontend direct      │
│  face-match           :8005  ← Backend (AML check)  │
│  capture-passport     :8006  ← Frontend direct      │
│  capture-face         :8007  ← Frontend direct      │
│                                                     │
│  MinIO :9000  |  PostgreSQL :5432  |  Redis :6379   │
└─────────────────────────────────────────────────────┘
```

---

## 4.5.2 Service de capture — capture-cin, capture-passport, capture-face

### Rôle

Ces trois services assurent la **capture guidée en temps réel** depuis la caméra du client. Ils intègrent un modèle YOLOv11 léger pour détecter la présence du document ou du visage dans le cadre avant de déclencher la capture automatique.

### Modèles embarqués

| Service | Modèle YOLO | Seuil confiance |
|---|---|---|
| `capture-cin` | `cin_recto_best.pt` + `cin_verso_.pt` | 0.40 (détection) / 0.60 (validation) |
| `capture-passport` | `passport_best.pt` | 0.40 / 0.60 |
| `capture-face` | `face_best.pt` | 0.40 (détection) / 0.60 (capture valide) |

### Endpoints principaux

#### WebSocket `GET /ws/detect` — Détection temps réel

Communication bidirectionnelle frame-par-frame pendant la session de capture.

**Requête (par frame) :**
```json
{
  "frame": "<base64_image>",
  "guide": { "x": 120, "y": 200, "width": 400, "height": 260 },
  "screen": { "width": 1280, "height": 720 }
}
```

**Réponse (par frame) :**
```json
{
  "status": "DETECTED",
  "confidence": 0.87,
  "is_inside_guide": true,
  "should_capture": false,
  "capture_id": null,
  "image_url": null
}
```

Les états possibles de `status` sont :
- `ABSENT` — aucun document détecté dans le cadre
- `DETECTED` — document détecté mais conditions non remplies (hors cadre, confiance insuffisante)
- `CONFIRMED` — 5 frames consécutives valides → déclenchement automatique de la capture

#### POST `/capture` — Capture manuelle

Déclenché par le bouton de capture du frontend.

**Requête :**
```json
{
  "frame": "<base64_image>",
  "capture_id": "session_abc123"
}
```
**Réponse :**
```json
{
  "success": true,
  "capture_id": "session_abc123",
  "image_path": "http://minio:9000/kyc-temp/captures/session_abc123.jpg"
}
```

#### POST `/capture-both` — Capture CIN recto + verso simultanée (capture-cin uniquement)

```json
{
  "recto": "<base64_recto>",
  "verso": "<base64_verso>",
  "session_id": "session_abc123"
}
```
**Réponse :**
```json
{
  "success": true,
  "recto_url": "http://minio:9000/kyc-temp/recto_abc123.jpg",
  "verso_url": "http://minio:9000/kyc-temp/verso_abc123.jpg",
  "session_id": "session_abc123"
}
```

Les images capturées sont **stockées sur MinIO** (`bucket: kyc-temp`) et leurs URLs sont transmises au backend pour alimenter les étapes suivantes du pipeline.

---

## 4.5.3 Service OCR — ocr-service (port 8003)

### Rôle

Le service OCR est le composant central du pipeline d'extraction. Il orchestre :
1. La détection des zones de champs par **YOLOv11m** (3 modèles : recto, verso, passeport)
2. La reconnaissance du texte par **PaddleOCR PP-OCRv5** (2 moteurs : arabe + français)
3. Le **post-traitement** des résultats (normalisation des dates, tri RTL des fragments arabes)

Les images ne sont pas envoyées directement au service : le **frontend transmet les URLs MinIO** des images préalablement capturées par les services de capture. Le service télécharge les images depuis MinIO, traite, puis retourne les données structurées au frontend.

### Endpoints

#### POST `/extract` — Extraction CIN (recto + verso)

**Requête :**
```json
{
  "minio_url_recto": "http://minio:9000/kyc-temp/recto_abc123.jpg",
  "minio_url_verso":  "http://minio:9000/kyc-temp/verso_abc123.jpg",
  "document_id": "doc_xyz789"
}
```

**Réponse :**
```json
{
  "document_id": "doc_xyz789",
  "structured": {
    "nom": "المعموري",
    "prenom": "ياسين",
    "nom_complet": "بن نورالدين بن البشير",
    "cin_number": "14447754",
    "date_naissance": "29 أفريل 2004",
    "lieu_naissance": "نابل",
    "adresse": "شارع الحبيب بورقيبة تونس",
    "profession": "طالب",
    "nom_mere": "فاطمة بنت محمد",
    "print_id": "TN12345",
    "issue_date": "15/03/2022"
  },
  "structured_raw": {
    "nom": "المعموري",
    "cin_number": "14447754",
    "date_naissance": "29افريل2004"
  },
  "raw_detections": [
    {
      "cin_type": "recto",
      "field": "num_cin",
      "yolo_score": 0.9901,
      "position": { "x1": 691, "y1": 390, "x2": 1162, "y2": 534 },
      "ocr_raw": "14447754",
      "ocr_text": "14447754",
      "ocr_score": 0.9999,
      "ocr_lines": [{ "text": "14447754", "score": 0.9999 }]
    }
  ],
  "confidence": 0.9413,
  "yolo_elapsed_ms": 620,
  "ocr_elapsed_ms": 1840,
  "face_crop_url": "http://minio:9000/kyc-temp/face_crop/doc_xyz789.png"
}
```

La réponse contient trois niveaux de données :
- `structured` : données post-traitées prêtes à l'utilisation
- `structured_raw` : données OCR brutes avant post-traitement (pour audit)
- `raw_detections` : détail complet de chaque champ (score YOLO, bbox, score OCR ligne par ligne)

Le service extrait et uploade automatiquement le **crop du visage** (`face_crop_url`) depuis la zone `image` du recto, utilisé ensuite par le service `face-match`.

#### POST `/extract-passport` — Extraction passeport

**Requête :**
```json
{
  "minio_url": "http://minio:9000/kyc-temp/passport_abc123.jpg",
  "document_id": "doc_xyz789"
}
```

**Réponse :**
```json
{
  "document_id": "doc_xyz789",
  "structured": {
    "last_name": "MAAMOURI",
    "first_name": "YASSINE",
    "full_name_ar": "بن نورالدين بن البشير",
    "nom_ar": "المعموري",
    "prenom_ar": "ياسين",
    "num_pass": "A12345678",
    "num_cin": "14447754",
    "dob": "29/04/2004",
    "pob": "نابل",
    "issue_date": "15/03/2022",
    "expiry_date": "14/03/2032",
    "address_ar": "شارع الحبيب بورقيبة تونس",
    "profession_ar": "طالب"
  },
  "structured_raw": { ... },
  "raw_detections": [ ... ],
  "confidence": 0.9375,
  "yolo_elapsed_ms": 6287,
  "ocr_elapsed_ms": 2100,
  "face_crop_url": "http://minio:9000/kyc-temp/face_crop/doc_xyz789.png"
}
```

#### GET `/health` — Supervision

```json
{
  "status": "ok",
  "service": "ocr-service",
  "port": 8003,
  "yolo_recto": true,
  "yolo_verso": true,
  "yolo_passport": true,
  "ocr_engine": true
}
```

### Initialisation lazy des modèles

Les modèles YOLO et PaddleOCR ne sont pas chargés au démarrage du conteneur mais lors du **premier appel** à chaque endpoint (lazy initialization). Un verrou threading (`_pipeline_lock`) sérialise les appels concurrents car YOLO et PaddleOCR ne sont pas thread-safe. Les appels suivants bénéficient des modèles déjà chargés en mémoire.

---

## 4.5.4 Service de comparaison biométrique — face-match (port 8005)

### Rôle

Compare le visage extrait du document d'identité (crop `face_crop_url` produit par l'OCR) avec le selfie capturé par `capture-face`, pour vérifier que la personne présentant le document est bien son titulaire.

### Endpoint

#### POST `/face-match`

**Requête :**
```json
{
  "cin_image_path": "http://minio:9000/kyc-temp/face_crop/doc_xyz789.png",
  "selfie_image_path": "http://minio:9000/kyc-temp/selfie_abc123.jpg",
  "tolerance": 0.6
}
```

**Réponse :**
```json
{
  "match": true,
  "similarity_score": 0.923,
  "distance": 0.077,
  "status": "SUCCESS",
  "message": "Visages correspondants"
}
```

Le paramètre `tolerance` définit le seuil de distance maximale acceptable (par défaut 0.6 — valeur standard pour la reconnaissance faciale). Un `similarity_score` > 0.85 est considéré comme un match fiable.

---

## 4.5.5 Format d'échange et stockage des images

### Stockage objet — MinIO

Toutes les images transitent via **MinIO** (compatible S3), jamais encodées en base64 dans les réponses JSON des services internes. Le flux est le suivant :

```
Frontend → capture service → MinIO (upload)     → URL MinIO retournée au frontend
Frontend → ocr-service/ocr-credit (URL MinIO)   → téléchargement MinIO → résultat JSON → frontend
Backend  → face-match (URL MinIO)               → comparaison biométrique → résultat → backend
```

Le bucket `kyc-temp` est utilisé pour les captures en cours de session. Les objets sont organisés par préfixe :

| Préfixe | Contenu |
|---|---|
| `captures/` | Captures brutes CIN et passeport |
| `face_crop/` | Crops visage extraits par l'OCR |
| `selfies/` | Photos selfie du client |

### Format des échanges inter-services

Tous les services communiquent en **JSON** sur HTTP/1.1. Les images sont référencées par URL MinIO, jamais transmises en binaire entre services internes. Les schémas Pydantic garantissent la validation des entrées/sorties à chaque frontière de service.

---

## 4.5.5 Service OCR crédit — ocr-credit (port 8004)

### Rôle

Le service `ocr-credit` est dédié à l'extraction de texte sur les **documents financiers** : attestations de travail et fiches de paie. Contrairement à l'`ocr-service` qui utilise YOLO pour localiser des champs précis, ce service applique directement **PaddleOCR PP-OCRv5 sur la totalité de l'image** sans étape de détection préalable. Cette approche est adaptée aux attestations dont la mise en page est variable selon les employeurs — il n'existe pas de structure fixe à détecter.

Le moteur tourne en **double langue simultanée** (arabe + français) pour couvrir les deux formats d'attestations en usage en Tunisie.

### Endpoint

#### POST `/extract-attestation`

**Requête :**
```json
{
  "document_id": "doc_credit_001",
  "minio_url": "http://minio:9000/kyc-temp/attestation_abc123.jpg"
}
```

**Réponse :**
```json
{
  "document_id": "doc_credit_001",
  "ar": {
    "text": "شهادة في الدخل صاحب العمل شركة XYZ الراتب الصافي 2500 دينار",
    "avg_score": 0.912,
    "lines": [
      { "text": "شهادة في الدخل", "score": 0.961, "bbox": {"x1": 120, "y1": 45, "x2": 480, "y2": 90} },
      { "text": "الراتب الصافي 2500 دينار", "score": 0.887, "bbox": {"x1": 95, "y1": 210, "x2": 510, "y2": 255} }
    ]
  },
  "fr": {
    "text": "ATTESTATION DE SALAIRE Employeur: Société XYZ Salaire net: 2500 DT",
    "avg_score": 0.945,
    "lines": [
      { "text": "ATTESTATION DE SALAIRE", "score": 0.981, "bbox": {"x1": 200, "y1": 40, "x2": 620, "y2": 85} },
      { "text": "Salaire net: 2500 DT", "score": 0.923, "bbox": {"x1": 90, "y1": 200, "x2": 450, "y2": 240} }
    ]
  },
  "ocr_elapsed_ms": 1240.5
}
```

La réponse retourne le texte brut reconnu **ligne par ligne** avec les coordonnées de chaque ligne (`bbox`) et les scores de confiance, séparément pour l'arabe (`ar`) et le français (`fr`). Le frontend transmet ensuite ce texte brut au backend lors de la soumission finale (`/submit-full`), qui se charge d'en extraire les champs utiles (salaire, employeur, date).

#### GET `/health`

```json
{
  "status": "ok",
  "service": "ocr-credit",
  "port": 8004,
  "ocr_engine": true
}
```

---

## 4.5.6 Contraintes de performance en production

### Temps de traitement mesurés

| Étape | Temps moyen | Notes |
|---|---|---|
| Détection YOLO capture (par frame) | ~17–20 ms | GPU, 640×640px |
| YOLO OCR recto | ~77 ms | GPU |
| YOLO OCR verso | ~5 300 ms | GPU, image haute résolution |
| YOLO OCR passeport | ~6 288 ms | GPU, image haute résolution |
| PaddleOCR par champ | ~150–300 ms | GPU, par crop |
| Pipeline OCR complet CIN | ~2 500 ms | recto + verso |
| Pipeline OCR passeport | ~8 400 ms | image unique |
| Face match | ~200–400 ms | selon résolution |
| OCR attestation (ocr-credit) | ~1 200 ms | double langue ar+fr, image complète |

### Contraintes mémoire

Le chargement simultané des modèles en mémoire GPU est la principale contrainte :

| Service | Modèles chargés | VRAM estimée |
|---|---|---|
| `capture-cin` | 2 × YOLOv11m | ~300 MB |
| `capture-passport` | 1 × YOLOv11m | ~150 MB |
| `capture-face` | 1 × YOLOv11 | ~100 MB |
| `ocr-service` | 3 × YOLOv11m + 4 × PaddleOCR PP-OCRv5 | ~3–4 GB |
| `ocr-credit` | 2 × PaddleOCR PP-OCRv5 (ar + fr) | ~1–2 GB |
| `face-match` | modèle embeddings | ~200 MB |

L'`ocr-service` est le plus gourmand. En production, une GPU dédiée de 8 GB minimum est recommandée pour ce service.

### Sérialisation des appels OCR

PaddleOCR et YOLO ne sont pas thread-safe. Le service OCR utilise un verrou global (`threading.Lock`) qui sérialise tous les appels `predict()`. En cas de charge importante, une file d'attente (Redis + Celery) peut être interposée entre le backend et l'`ocr-service` pour absorber les pics.

### Endpoints de supervision

Chaque service expose un `GET /health` qui retourne l'état de chargement de ses modèles. Ces endpoints peuvent être interrogés au démarrage pour vérifier que tous les composants IA sont prêts avant d'accepter des demandes clients.

```json
{
  "status": "ok",
  "service": "ocr-service",
  "yolo_recto": true,
  "yolo_verso": true,
  "yolo_passport": true,
  "ocr_engine": true
}
```

---

## 4.5.7 Flux complet d'une demande KYC

Le parcours KYC se déroule en deux phases distinctes : une **phase de collecte** entièrement pilotée par le frontend, suivie d'une **phase de validation** déclenchée par le backend lors de la soumission finale.

```
─── PHASE COLLECTE (Frontend → services IA directs) ──────────────────────

1. Client ouvre la caméra CIN
        │
        ▼
2. Frontend → capture-cin :8001  [WebSocket /ws/detect]
   → Détection YOLO en temps réel frame par frame
   → "CONFIRMED" après 5 frames valides
   → Upload recto + verso → MinIO
   → Retourne recto_url + verso_url au Frontend
        │
        ▼
3. Frontend → ocr-service :8003  [POST /extract]
   → Transmet recto_url + verso_url (URLs MinIO)
   → ocr-service télécharge les images depuis MinIO
   → YOLO localise les champs → PaddleOCR lit le texte
   → Post-traitement (dates, RTL)
   → Upload face_crop → MinIO
   → Retourne JSON structuré + face_crop_url au Frontend
        │
        ▼
4. Frontend → capture-passport :8006  [WebSocket /ws/detect]  (si passeport)
   OU
   Frontend → capture-face :8007  [POST /capture]
   → Client prend son selfie
   → Upload selfie → MinIO
   → Retourne selfie_url au Frontend
        │
        ▼
5. Frontend POST /submit-full → Backend :8000
   → Transmet : recto_url, verso_url, selfie_url, face_crop_url,
                données OCR structurées, session_id

─── PHASE VALIDATION (Backend → face-match uniquement) ───────────────────

        │
        ▼
6. Backend → face-match :8005  [POST /face-match]
   → Transmet face_crop_url + selfie_url
   → Comparaison biométrique inter-clients (filtrage AML)
   → Retourne similarity_score au Backend
        │
        ▼
7. Backend → logique de validation (scoring de risque)
   → Décision : LOW / MEDIUM / HIGH
   → Stockage PostgreSQL (pipeline + documents)
   → Résultat final → Frontend
        │
        ▼
8. Admin → Backend [POST /validate-kyc]
   → Revue manuelle du dossier
   → Approbation / Rejet
   → Notification client
```

Cette séparation des responsabilités a deux avantages : le backend ne traite jamais d'images binaires (il ne reçoit que des URLs et des données JSON), et les services IA de capture et d'OCR peuvent évoluer indépendamment sans modifier la logique métier du backend.
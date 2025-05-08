# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 01:10:11 2025

@author: andy
"""

import cv2
import torch
import numpy as np
import os
import glob
import time
from ultralytics import YOLO
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn.functional as F
import timm
import faiss
from collections import defaultdict  # <--- Für label_to_embeddings

# -----------------------------------------------------------------------------
# Geräte-Auswahl & allgemeine Settings
# -----------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# YOLOv8s laden (für die Objekterkennung)
model = YOLO('yolov8s.pt')  

# TIMM-Modell laden – z.B. convnextv2_atto
timm_model = timm.create_model('convnext_tiny_in22ft1k', pretrained=False, num_classes=62)
timm_model.load_state_dict(torch.load("trained_convtiny_v02.pth"))



timm_model.eval().to(device)

# -----------------------------------------------------------------------------
# Hyperparameter
# -----------------------------------------------------------------------------
SIMILARITY_THRESHOLD = 0.2484  # Schwellwert, ab wann das Objekt als "Match" gilt
TOP_percentage = 40           # Prozent der Werte im Feauture Vektor, die ausgewertet werden
# 40 -> 0.349    50 -> 0.41    60 -> 0.44    75 ->  0.49   100 -> 0.52
# 40 -> xx       50 -> xx     60 ->  0.36    75 ->  0.46   100 -> 0.52
YOLO_THRESHOLD = 0.004
YOLO_MaxAnzahl = 256
YOLO_IOU = 0.2

IMG_SIZE = 224  # Zielgröße der Crops (Breite/Höhe) für das Timm-Modell

# Mittelwerte und Standardabweichungen (TIMM-Standard)
MEAN = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)

# -----------------------------------------------------------------------------
# Hilfsfunktionen
# -----------------------------------------------------------------------------
def l2_normalize(vec):
    """Normiert einen Numpy-Vektor auf Länge 1."""
    norm = np.linalg.norm(vec)
    return vec / norm if norm != 0 else vec

def prepare_crop_torch(crop_bgr, device):
    """
    crop_bgr: np.array (H, W, 3), dtype=uint8, OpenCV-Format (BGR)
    device:   torch.device

    Gibt einen Torch-Tensor mit Shape (3, IMG_SIZE, IMG_SIZE) zurück,
    der bereits normalisiert (mean/std) und in RGB ist.
    """
    # 1) In Torch-Tensor konvertieren (float)
    tensor = torch.from_numpy(crop_bgr).float()  # => (H, W, C)

    # 2) Kanal-Reihenfolge: BGR -> RGB
    tensor = tensor.permute(2, 0, 1)  # => (C, H, W)
    tensor = tensor[[2, 1, 0], ...]   # Kanäle tauschen

    # 3) Auf das gewünschte Gerät schieben
    tensor = tensor.to(device)

    # 4) Resize auf (IMG_SIZE, IMG_SIZE)
    tensor = F.interpolate(tensor.unsqueeze(0), size=(IMG_SIZE, IMG_SIZE),
                           mode='bilinear', align_corners=False)
    tensor = tensor.squeeze(0)  # Zurück zu (3, IMG_SIZE, IMG_SIZE)

    # 5) Normalisieren (im Bereich 0..255 -> 0..1)
    tensor = tensor / 255.0
    tensor = (tensor - MEAN) / STD

    return tensor

def extract_embedding_timm_torch(image_bgr):
    """
    Nimmt ein OpenCV-BGR-Bild (np.array) und gibt ein CPU-Numpy-Embedding zurück.
    """
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
        # Preprocessing in Torch
        t = prepare_crop_torch(image_bgr, device)
        # Batch-Dimension anfügen
        input_batch = t.unsqueeze(0)  # => (1, 3, H, W)

        # Durch das Modell jagen
        features = timm_model.forward_features(input_batch)
        # Poolen (falls 4D)
        if features.ndim == 4:
            embeddings = F.adaptive_avg_pool2d(features, 1).squeeze(-1).squeeze(-1)
                         
        elif features.ndim == 3:
            embeddings = features.mean(dim=1)
        else:
            raise ValueError("Unsupported embedding shape")

    # Embedding zurück auf CPU + Numpy
    emb = embeddings.cpu().numpy().reshape(-1)
    return emb

# -----------------------------------------------------------------------------
# Schritt 1: Referenzbilder laden & Embeddings berechnen
# -----------------------------------------------------------------------------
reference_dirs = [
    ('Cornflakes', 'C:/python/Objekt_finder/Cornflakes/'),
    ('Biomais', 'C:/python/Objekt_finder/Biomais/'),
    ('Handschuhe', 'C:/python/Objekt_finder/Handschuhe/'),
    ('Tomatensosse', 'C:/python/Objekt_finder/Tomatensosse/'),
    ('Biosenf', 'C:/python/Objekt_finder/Biosenf/'),
    ('Dinkelstangen', 'C:/python/Objekt_finder/Dinkelstangen/'),
    ('Apfelmark', 'C:/python/Objekt_finder/Apfelmark/')
]

reference_embeddings_list = []
reference_labels = []

# Dictionary: label → Liste von Embeddings
label_to_embeddings = defaultdict(list)

for label, reference_dir in reference_dirs:
    for file in os.listdir(reference_dir):
        img_path = os.path.join(reference_dir, file)
        img = cv2.imread(img_path)
        if img is None:
            continue
        emb = extract_embedding_timm_torch(img)
        emb = l2_normalize(emb)
        reference_embeddings_list.append(emb)
        reference_labels.append(label)
        label_to_embeddings[label].append(emb)

reference_embeddings = np.array(reference_embeddings_list, dtype='float32')
reference_embeddings /= (np.linalg.norm(reference_embeddings, axis=1, keepdims=True) + 1e-8)

# -----------------------------------------------------------------------------
# Schritt 2: Ermittlung der Top 50 % der Feature-Dimensionen nach Varianz
# -----------------------------------------------------------------------------

# 1) Klassendurchschnitte (Mean pro Label)
class_means = {}
for lbl, embs in label_to_embeddings.items():
    embs = np.array(embs, dtype='float32')  # shape: (num_samples_in_label, embedding_dim)
    class_means[lbl] = np.mean(embs, axis=0)  # shape: (embedding_dim,)

# 2) Inter-Klassen-Varianz pro Dimension
#    Wir betrachten alle Klassenmittelwerte und berechnen deren Varianz über die Labels.
mean_matrix = np.array(list(class_means.values()))  # shape: (num_classes, embedding_dim)
inter_class_variance = np.var(mean_matrix, axis=0)  # shape: (embedding_dim,)

# 3) Intra-Klassen-Varianz pro Dimension (durchschnittlich über alle Klassen)
#    Für jede Klasse berechnen wir die Varianz ihrer Samples pro Dimension,
#    und mitteln das anschließend über alle Klassen.
num_classes = len(class_means)
embedding_dim = reference_embeddings.shape[1]
intra_class_variance = np.zeros(embedding_dim, dtype=np.float32)

for lbl, embs in label_to_embeddings.items():
    e = np.array(embs, dtype='float32')  # shape: (num_samples_in_label, embedding_dim)
    var_dims = np.var(e, axis=0)         # shape: (embedding_dim,)
    intra_class_variance += var_dims

intra_class_variance /= num_classes  # Mittelwert über alle Klassen

# 4) Ratio aus Inter-/Intra-Varianz
#    => Je größer die Ratio, desto besser trennt das Feature die Klassen.
ratio = inter_class_variance / (intra_class_variance + 1e-8)  # shape: (embedding_dim,)

# 5) Top-Features auswählen (z.B. Top 50%)
num_features_to_keep = embedding_dim*TOP_percentage // 100  # 30%
# => Sortierung nach Ratio, größtes zuerst
top_feature_indices = np.argsort(ratio)[-num_features_to_keep:]

# 6) Reduziere Referenz-Embeddings auf diese diskriminativsten Dimensionen
reference_embeddings_reduced = reference_embeddings[:, top_feature_indices]

# 7) FAISS-Index aufbauen
faiss_index = faiss.IndexFlatIP(reference_embeddings_reduced.shape[1])
faiss_index.add(reference_embeddings_reduced)


# -----------------------------------------------------------------------------
# Schritt 3: Video-Loop mit YOLO-Detection und Embedding-Check
# -----------------------------------------------------------------------------
cap = cv2.VideoCapture(0)  # Standard-Kamera; ggf. Index anpassen
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

video_path = 'dm1.mp4'
cap2 = cv2.VideoCapture(video_path)

if not cap2.isOpened():
    print("Fehler beim Öffnen der Videodatei")
    exit()

print("Starte Echtzeit-Erkennung mit TIMM & YOLO...")

# Timer-Variablen
total_frames = 0
yolo_total_time = 0.0
conv_total_time = 0.0
full_total_time = 0.0

while True:
    ret, frameraw = cap2.read()
    if not ret:
        break

    # Frame um 90° nach rechts drehen (im Uhrzeigersinn)
    frame = cv2.rotate(frameraw, cv2.ROTATE_90_CLOCKWISE)

    # Für Benchmark: Ein Standardbild laden (kannst du anpassen)
    #frame = cv2.resize(cv2.imread('20250315_185048.jpg'), (1920, 2560))
    
    start_full = time.time()  # Gesamt-Timer Start
    start_yolo = time.time()  # YOLO-Timer Start

    # YOLO-Detection
    results = model(frame, imgsz=960, conf=YOLO_THRESHOLD,
                    iou=YOLO_IOU, max_det=YOLO_MaxAnzahl)
    yolo_total_time += time.time() - start_yolo

    # Für alle gefundenen Objekte
    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy()

        crop_tensors = []
        crop_infos = []

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)

            # a) Person (COCO class ID 0) überspringen
            if int(classes[i]) == 0:
                continue

            # b) Sehr große Objekte überspringen
            bbox_area = (x2 - x1) * (y2 - y1)
            image_area = frame.shape[0] * frame.shape[1]
            if bbox_area / image_area > 0.2:
                continue

            # Kleiner Puffer (10px) am Rand
            x1b = max(x1 - 10, 0)
            y1b = max(y1 - 10, 0)
            x2b = min(x2 + 10, frame.shape[1])
            y2b = min(y2 + 10, frame.shape[0])

            crop = frame[y1b:y2b, x1b:x2b]
            if crop.size == 0:
                continue

            # Crop per Torch-Pipeline vorbereiten:
            t = prepare_crop_torch(crop, device)
            crop_tensors.append(t)
            crop_infos.append((x1b, y1b, x2b, y2b))

        if not crop_tensors:
            continue

        # CNN-Forward in einem Rutsch (Batch-Verarbeitung)
        start_conv = time.time()
        input_batch = torch.stack(crop_tensors, dim=0)  # => (B, 3, 224, 224)

        with torch.cuda.amp.autocast(dtype=torch.float16), torch.no_grad():
            features = timm_model.forward_features(input_batch)
            if features.ndim == 4:
                embeddings = F.adaptive_avg_pool2d(features, 1).squeeze(-1).squeeze(-1)
            elif features.ndim == 3:
                embeddings = features.mean(dim=1)
            else:
                raise ValueError("Unsupported embedding shape")

        # Embeddings auf CPU -> Numpy
        embeddings = embeddings.cpu().numpy()

        # L2-Normalisierung
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-8)

        conv_total_time += time.time() - start_conv

        # *** Reduktion auf Top-Features ***
        embeddings_reduced = embeddings[:, top_feature_indices].astype('float32')

        # Faiss-Suche => Bei IndexFlatIP entspricht Score dem Cosinus (da normalisiert)
        D, I = faiss_index.search(embeddings_reduced, k=1)
        # D.shape = (B,1), I.shape = (B,1)

        # Label-Zuordnung
        matched_labels = [reference_labels[idx] for idx in I[:, 0]]

        # Zeichnen
        for sim, label, (x1b, y1b, x2b, y2b) in zip(D[:, 0], matched_labels, crop_infos):
            if sim > SIMILARITY_THRESHOLD:
                cv2.rectangle(frame, (x1b, y1b), (x2b, y2b), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {sim:.2f}", (x1b, y1b - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                cv2.rectangle(frame, (x1b, y1b), (x2b, y2b), (255, 0, 0), 1)
                cv2.putText(frame, f"nom {sim:.2f}", (x1b, y1b - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    total_frames += 1
    full_total_time += time.time() - start_full

    cv2.imshow("YOLOv8 + TIMM Matching", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# -----------------------------------------------------------------------------
# Benchmark-Ausgabe
# -----------------------------------------------------------------------------
print("\n--- Benchmark-Resultate ---")
print(f"Gesamt-Frames verarbeitet: {total_frames}")
print(f"Durchschnittliche YOLO-Zeit pro Frame: {yolo_total_time / total_frames:.3f} Sekunden")
print(f"Durchschnittliche Conv-Zeit pro Frame: {conv_total_time / total_frames:.3f} Sekunden")
print(f"Durchschnittliche Gesamtzeit pro Frame: {full_total_time / total_frames:.3f} Sekunden")

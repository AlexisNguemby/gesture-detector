import urllib.request
import os

# URLs possibles du modèle
URLS = [
    "https://storage.googleapis.com/mediapipe-assets/hand_landmarker.task",
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker.task",
    "https://github.com/google-ai-edge/mediapipe/raw/master/mediapipe/tasks/python/test_data/hand_landmarker.task"
]

MODEL_PATH = "hand_landmarker.task"

for url in URLS:
    print(f"Tentative de téléchargement depuis {url}...")
    try:
        urllib.request.urlretrieve(url, MODEL_PATH)
        print(f"✓ Modèle téléchargé avec succès: {MODEL_PATH}")
        break
    except Exception as e:
        print(f"  ✗ Erreur: {e}")
else:
    print("\n⚠ Impossible de télécharger le modèle. Consultez:")
    print("  https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/hands.md")


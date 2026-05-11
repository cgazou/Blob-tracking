Blob Tracker
============

Lancement
---------

### Version carrée (blob_tracker_app_square.py)

```bash

python blob_tracker_app_square.py --input video.mp4
```
### Version rectangulaire (blob_tracker_app_rectangle.py)
```
bash

python blob_tracker_app_rectangle.py --input video.mp4
```
Options principales
-------------------

| Option | Description |
| --- | --- |
| `--input` | Fichier vidéo ou `0` pour webcam |
| `--output` | Sauvegarde en MP4 |
| `--png-output` | Dossier pour export PNG avec transparence |
| `--threshold` | Seuil manuel (0-255) |
| `--min-area` | Surface minimale des blobs |
| `--max-area` | Surface maximale des blobs |
| `--max-blobs` | Nombre max de blobs à suivre |
| `--mask-input` | Dossier avec masques PNG (canal alpha) |

Raccourcis clavier (fenêtre de preview)
---------------------------------------

| Touche | Action |
| --- | --- |
| `q` | Quitter |
| `t` | Trails (traces) |
| `c` | Connections |
| `b` | Brackets (crochets) |
| `m` | Métriques |
| `g` | Grille |
| `d` | Pointillés |
| `x` | Boîtes |
| `p` | Point central (off → dot → cross) |

Configuration
-------------

Créer un fichier `config.txt` dans le même dossier :
```
text

THRESHOLD_VALUE = 127
MIN_AREA = 10000
MAX_AREA = 100000
MAX_BLOBS = 2
BLOB_NAMES = Rouge,Vert,Bleu,Jaune
```

Export PNG avec alpha (fond transparent)
----------------------------------------
```
bash

python blob_tracker_app_rectangle.py --input video.mp4 --png-output ./alpha_frames
```

Les PNG exportés ont un fond transparent (canal alpha) prêt pour compositing.
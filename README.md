# Hand Talk Translator

TFE ECAM — webcam → reconnaissance de signes (LSTM) → phrase en français + voix.  
Il y a aussi un pipeline gant ESP32 (`firmware/`, scripts `glove_*`).

## Install

```powershell
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Si PowerShell bloque l’activation :  
`Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`

## Lancer

```powershell
py -3.11 main.py
```

Sans webcam (replay) : `py -3.11 main.py --replay sessions/demo.jsonl`

Tests : `py -3.11 -m pytest -v`

## Entraîner (vision)

```powershell
py -3.11 collect_data.py
py -3.11 train_model.py
py -3.11 evaluate_model.py
```

Données : `data/` — modèles : `models/`

## Gant

```powershell
py -3.11 glove_calibration.py --port COM5
py -3.11 glove_collect.py --port COM5
py -3.11 glove_train.py
py -3.11 glove_recognize.py --port COM5
```

Sans matériel : `--port MOCK`

## TFE

- `docs/dossier_ia/` — transparence IA ([GitHub](https://github.com/Prota0202/Hand-Talk-Translator/tree/main/docs/dossier_ia))

Rapport PDF, déclaration signée et CDC : remis au jury hors dépôt (archive locale).

Réglages : `config.py`

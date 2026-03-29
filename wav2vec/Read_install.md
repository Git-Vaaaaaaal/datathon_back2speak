Oui. Sur **Windows 11**, fais ça dans **PowerShell**.

## 1. Aller dans ton dossier projet

```powershell
cd D:\Dataton2026
mkdir pronunciation_analyzer
cd pronunciation_analyzer
```

## 2. Créer l’environnement virtuel

Avec Python 3.10 :

```powershell
py -3.10 -m venv .venv
```

## 3. Activer l’environnement

```powershell
.venv\Scripts\Activate.ps1
```

Si PowerShell bloque l’activation, lance une fois :

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

puis relance :

```powershell
.venv\Scripts\Activate.ps1
```

## 4. Mettre à jour pip

```powershell
python -m pip install --upgrade pip setuptools wheel
```

## 5. Installer les dépendances

```powershell
pip install fastapi uvicorn soundfile numpy torch transformers phonemizer python-multipart
```

## 6. Vérifier les versions installées

```powershell
python -c "import sys; print(sys.executable)"
python -c "import numpy, torch, transformers, fastapi, soundfile; print('OK')"
```

## 7. Installer eSpeak NG côté Windows

Le programme en a besoin pour `phonemizer`.

Installe **eSpeak NG** sur Windows, idéalement dans :

```text
C:\Program Files\eSpeak NG
```

Le dossier doit contenir au moins :

* `espeak-ng.exe`
* `libespeak-ng.dll`

## 8. Lancer l’API

Quand ton `main.py` est prêt :

```powershell
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

Puis ouvre :

```text
http://127.0.0.1:8000/docs
```

## Version courte en bloc

```powershell
cd D:\Dataton2026
mkdir pronunciation_analyzer
cd pronunciation_analyzer
py -3.10 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install fastapi uvicorn soundfile numpy torch transformers phonemizer python-multipart
```

Si tu veux, je peux aussi te donner un `requirements.txt` prêt à copier-coller.

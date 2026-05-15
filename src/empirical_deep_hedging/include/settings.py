import json
from pathlib import Path


SETTINGS_DIR = Path("settings")


def _settings_path(name):
    direct = SETTINGS_DIR / f"{name}.json"
    if direct.exists():
        return direct

    matches = sorted(SETTINGS_DIR.glob(f"*/{name}.json"))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"Could not find settings file for {name}.")

    match_list = ", ".join(str(path) for path in matches)
    raise FileNotFoundError(f"Ambiguous settings file for {name}: {match_list}.")


def _settings_output_path(name):
    SETTINGS_DIR.mkdir(exist_ok=True)
    for child in sorted(SETTINGS_DIR.iterdir()):
        if child.is_dir() and name.startswith(child.name):
            return child / f"{name}.json"
    return SETTINGS_DIR / f"{name}.json"

class Settings():
    def __init__(self):
        with open('settings.json') as f:
            self.data = json.load(f)
        
    def save(self, name):
        path = _settings_output_path(name)
        with path.open('w') as f:
            json.dump(self.data, f)
            
    def load(self, name):
        path = _settings_path(name)
        with path.open('r') as f:
            self.data = json.load(f)

s = Settings()

def getSettings():
    return s.data

def setSettings(fname):
    s.load(fname)

def saveSettings(fname):
    s.save(fname)

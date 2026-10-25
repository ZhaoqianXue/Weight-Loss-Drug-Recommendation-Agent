"""Resolve explicit configuration independently of the process working directory."""
from dataclasses import dataclass
from functools import lru_cache
import json
import os
from pathlib import Path

@dataclass(frozen=True)
class Settings:
    root: Path
    config_path: Path
    config: dict

    def path(self, key):
        value = Path(self.config[key]).expanduser()
        return value if value.is_absolute() else self.root / value

    def require_writable(self):
        if self.config.get('frozen', True):
            raise ValueError('This run is frozen. Create a new run with weightloss new-run --run-id NAME.')

    @property
    def raw(self):
        return self.path('raw_dir') / 'webmd_all_reviews.csv'

    @property
    def collection_manifest(self):
        return self.path('raw_dir') / 'collection_manifest.json'

    @property
    def dataset_manifest(self):
        return self.path('standardized').parent / 'dataset_manifest.json'

@lru_cache(maxsize=1)
def get_settings():
    configured_root = os.environ.get('WEIGHTLOSS_ROOT')
    configured_file = os.environ.get('WEIGHTLOSS_CONFIG')
    if configured_root:
        root = Path(configured_root).expanduser().resolve()
    elif configured_file:
        # External configs should declare a project_root or use WEIGHTLOSS_ROOT.
        candidate = Path(configured_file).expanduser().resolve()
        payload = json.loads(candidate.read_text())
        if 'project_root' in payload:
            base = Path(payload['project_root']).expanduser()
            root = (candidate.parent / base).resolve() if not base.is_absolute() else base.resolve()
        else:
            raise ValueError('Set WEIGHTLOSS_ROOT or project_root for an external configuration.')
    else:
        root = next((p for p in Path(__file__).resolve().parents if (p/'configs/pipeline.json').is_file()), None)
        if root is None:
            raise ValueError('Set WEIGHTLOSS_ROOT to the research checkout (data are not shipped in the wheel).')
    config_path = Path(configured_file).expanduser() if configured_file else root/'configs/pipeline.json'
    if not config_path.is_absolute():
        config_path = root/config_path
    config_path = config_path.resolve()
    config = json.loads(config_path.read_text())
    if config.get('schema_version') != 1:
        raise ValueError('Unsupported pipeline configuration schema')
    return Settings(root, config_path, config)

def configure(root=None, config=None):
    if root is not None:
        os.environ['WEIGHTLOSS_ROOT'] = str(Path(root).expanduser().resolve())
    if config is not None:
        os.environ['WEIGHTLOSS_CONFIG'] = str(Path(config).expanduser().resolve())
    get_settings.cache_clear()

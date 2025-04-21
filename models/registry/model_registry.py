import os
import json
import shutil
import threading
from pathlib import Path
from typing import Optional, Dict, Any, List, Union

class ModelRegistry:
    def __init__(self, registry_path: Union[str, Path]):
        self.registry_path = Path(registry_path)
        self.index_file = self.registry_path / "registry_index.json"
        self.lock = threading.RLock()
        self.index = {"models": {}}
        self._load_index()

    def _acquire_lock(self, timeout: Optional[float] = None):
        got_lock = self.lock.acquire(timeout=timeout) if timeout is not None else self.lock.acquire()
        if not got_lock:
            raise TimeoutError("Could not acquire registry lock within timeout")

    def _release_lock(self):
        if self.lock._is_owned():
            self.lock.release()

    def _load_index(self):
        self.registry_path.mkdir(parents=True, exist_ok=True)
        if self.index_file.exists():
            try:
                with open(self.index_file, "r") as f:
                    self.index = json.load(f)
            except Exception:
                self.index = {"models": {}}
        else:
            self._save_index()

    def _save_index(self):
        try:
            with open(self.index_file, "w") as f:
                json.dump(self.index, f, indent=2)
        except Exception:
            pass

    def register_model(
        self,
        model_name: str,
        model_path: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None,
        copy_model: bool = True,
        version: Optional[str] = None,
        timeout: Optional[float] = 5.0,
        batch_metadata: Optional[List[Dict[str, Any]]] = None,
        batch_performance: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        self._acquire_lock(timeout)
        try:
            if model_name not in self.index["models"]:
                self.index["models"][model_name] = {"versions": {}}

            if version is None:
                version = self._generate_version(model_name)
            if version in self.index["models"][model_name]["versions"]:
                raise ValueError(f"Version {version} already exists for model {model_name}")

            model_dir = self.registry_path / model_name / version
            model_dir.mkdir(parents=True, exist_ok=True)

            model_file = Path(model_path)
            dest_file = model_dir / model_file.name
            if copy_model:
                shutil.copy2(model_file, dest_file)
            else:
                shutil.move(model_file, dest_file)

            version_meta = metadata.copy() if metadata else {}
            if batch_metadata:
                version_meta["batch_metadata"] = batch_metadata
            if batch_performance:
                version_meta["batch_performance"] = batch_performance

            self.index["models"][model_name]["versions"][version] = {
                "model_file": str(dest_file),
                "metadata": version_meta,
            }
            self._save_index()
            return version
        finally:
            self._release_lock()

    def get_model(self, model_name: str, version: Optional[str] = None):
        self._acquire_lock()
        try:
            if model_name not in self.index["models"]:
                raise ValueError(f"Model {model_name} not found")
            if version is None:
                version = self.get_latest_version(model_name)
            if version not in self.index["models"][model_name]["versions"]:
                raise ValueError(f"Version {version} not found for model {model_name}")
            model_file = self.index["models"][model_name]["versions"][version]["model_file"]
            metadata = self.index["models"][model_name]["versions"][version]["metadata"]
            return model_file, metadata
        finally:
            self._release_lock()

    def get_latest_version(self, model_name: str) -> str:
        self._acquire_lock()
        try:
            versions = self.list_versions(model_name)
            if not versions:
                raise ValueError(f"No versions found for model {model_name}")
            # Assume semantic versioning or numeric
            return sorted(versions, key=lambda v: [int(x) if x.isdigit() else x for x in v.split('.')])[-1]
        finally:
            self._release_lock()

    def list_models(self) -> List[str]:
        self._acquire_lock()
        try:
            return list(self.index["models"].keys())
        finally:
            self._release_lock()

    def list_versions(self, model_name: str) -> List[str]:
        self._acquire_lock()
        try:
            if model_name not in self.index["models"]:
                raise ValueError(f"Model {model_name} not found")
            return list(self.index["models"][model_name]["versions"].keys())
        finally:
            self._release_lock()

    def get_metadata(self, model_name: str, version: str) -> Dict[str, Any]:
        self._acquire_lock()
        try:
            if model_name not in self.index["models"]:
                raise ValueError(f"Model {model_name} not found")
            if version not in self.index["models"][model_name]["versions"]:
                raise ValueError(f"Version {version} not found for model {model_name}")
            return self.index["models"][model_name]["versions"][version]["metadata"]
        finally:
            self._release_lock()

    def compare_versions(self, model_name: str, v1: str, v2: str) -> Optional[Dict[str, Any]]:
        self._acquire_lock()
        try:
            meta1 = self.get_metadata(model_name, v1)
            meta2 = self.get_metadata(model_name, v2)
            return {"v1": meta1, "v2": meta2}
        finally:
            self._release_lock()

    def delete_version(self, model_name: str, version: str):
        self._acquire_lock()
        try:
            if model_name not in self.index["models"]:
                raise ValueError(f"Model {model_name} not found")
            if version not in self.index["models"][model_name]["versions"]:
                raise ValueError(f"Version {version} not found for model {model_name}")

            model_dir = self.registry_path / model_name / version
            if model_dir.exists():
                shutil.rmtree(model_dir)
            del self.index["models"][model_name]["versions"][version]
            if not self.index["models"][model_name]["versions"]:
                del self.index["models"][model_name]
                model_root = self.registry_path / model_name
                if model_root.exists() and not any(model_root.iterdir()):
                    model_root.rmdir()
            self._save_index()
        finally:
            self._release_lock()

    def delete_model(self, model_name: str):
        self._acquire_lock()
        try:
            if model_name not in self.index["models"]:
                raise ValueError(f"Model {model_name} not found")
            model_root = self.registry_path / model_name
            if model_root.exists():
                shutil.rmtree(model_root)
            del self.index["models"][model_name]
            self._save_index()
        finally:
            self._release_lock()

    def export_model(self, model_name: str, version: str, export_path: Union[str, Path]) -> str:
        self._acquire_lock()
        try:
            model_file, _ = self.get_model(model_name, version)
            export_path = Path(export_path)
            export_path.mkdir(parents=True, exist_ok=True)
            dest_file = export_path / Path(model_file).name
            shutil.copy2(model_file, dest_file)
            return str(dest_file)
        finally:
            self._release_lock()

    def _generate_version(self, model_name: str) -> str:
        versions = self.get_versions(model_name)
        if not versions:
            return "1.0.0"
        parts = versions[-1].split('.')
        if parts and parts[-1].isdigit():
            parts[-1] = str(int(parts[-1]) + 1)
            return '.'.join(parts)
        return f"{versions[-1]}_new"

    def get_versions(self, model_name: str) -> List[str]:
        return self.list_versions(model_name)

    def compare_batch_metrics(self, model_name: str, version: str) -> Optional[List[Dict[str, Any]]]:
        self._acquire_lock()
        try:
            meta = self.get_metadata(model_name, version)
            return meta.get("batch_performance")
        finally:
            self._release_lock()


    # At the bottom of models/registry/model_registry.py
    
_registry_instance = None
    
def get_registry(registry_path: Union[str, Path] = None) -> ModelRegistry:
    """
    Returns a singleton instance of ModelRegistry.
    If registry_path is not provided, defaults to '/tmp/models'.
    """
    global _registry_instance
    if _registry_instance is None:
        if registry_path is None:
            registry_path = '/tmp/models'
        _registry_instance = ModelRegistry(registry_path=registry_path)
    return _registry_instance

# Backward compatibility: module-level singleton
registry = get_registry()

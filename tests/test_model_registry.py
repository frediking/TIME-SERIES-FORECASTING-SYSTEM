import os
import shutil
import tempfile
import pytest
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from models.registry.model_registry import ModelRegistry

# Dummy model class for testing (replace with your actual model class)
class XGBoostModel:
    def __init__(self, name="test_model"):
        self.name = name
        self.metadata = {}
        self._fitted = False

    def fit(self, X, y):
        self._fitted = True

    def save(self, path):
        with open(path, "wb") as f:
            f.write(b"dummy model data")

@pytest.fixture
def test_model():
    """Factory for creating independent test models (thread-safe)."""
    def make_model():
        model = XGBoostModel(name="test_model")
        X = pd.DataFrame({
            'feature1': np.random.rand(100),
            'feature2': np.random.rand(100)
        })
        y = np.random.rand(100)
        model.fit(X, y)
        model.metadata.update({
            'test_metadata': 'test_value',
            'metrics': {
                'RMSE': 0.1,
                'MAE': 0.05
            }
        })
        return model
    return make_model

@pytest.fixture
def temp_model_dir():
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def registry(tmp_path):
    return ModelRegistry(registry_path=tmp_path)

class TestModelRegistry:

    def test_register_model(self, registry, test_model, temp_model_dir):
        model = test_model()
        model_path = os.path.join(temp_model_dir, 'test_model.pkl')
        model.save(model_path)
        version = registry.register_model(
            model_name="test_xgboost",
            model_path=model_path,
            metadata=model.metadata,
            version="1.0.0"
        )
        assert version == "1.0.0"
        assert "test_xgboost" in registry.list_models()
        assert "1.0.0" in registry.list_versions("test_xgboost")

    def test_get_model(self, registry, test_model, temp_model_dir):
        model = test_model()
        model_path = os.path.join(temp_model_dir, 'test_model.pkl')
        model.save(model_path)
        registry.register_model(
            model_name="test_xgboost",
            model_path=model_path,
            metadata=model.metadata,
            version="1.0.0"
        )
        model_file, metadata = registry.get_model("test_xgboost", "1.0.0")
        assert os.path.exists(model_file)
        assert metadata['test_metadata'] == 'test_value'

    def test_get_latest_version(self, registry, test_model, temp_model_dir):
        model = test_model()
        model_path = os.path.join(temp_model_dir, 'test_model.pkl')
        model.save(model_path)
        registry.register_model(
            model_name="test_xgboost",
            model_path=model_path,
            metadata=model.metadata,
            version="1.0.0"
        )
        registry.register_model(
            model_name="test_xgboost",
            model_path=model_path,
            metadata=model.metadata,
            version="1.1.0"
        )
        latest_version = registry.get_latest_version("test_xgboost")
        assert latest_version == "1.1.0"
        model_file, metadata = registry.get_model("test_xgboost", version=None)
        assert "1.1.0" in str(model_file)

    def test_list_models(self, registry, test_model, temp_model_dir):
        model = test_model()
        model_path = os.path.join(temp_model_dir, 'test_model.pkl')
        model.save(model_path)
        registry.register_model(
            model_name="test_xgboost1",
            model_path=model_path,
            metadata=model.metadata,
            version="1.0.0"
        )
        models = registry.list_models()
        assert "test_xgboost1" in models

    def test_list_versions(self, registry, test_model, temp_model_dir):
        model = test_model()
        model_path = os.path.join(temp_model_dir, 'test_model.pkl')
        model.save(model_path)
        registry.register_model(
            model_name="test_xgboost",
            model_path=model_path,
            metadata=model.metadata,
            version="1.0.0"
        )
        versions = registry.list_versions("test_xgboost")
        assert "1.0.0" in versions

    def test_get_metadata(self, registry, test_model, temp_model_dir):
        model = test_model()
        model_path = os.path.join(temp_model_dir, 'test_model.pkl')
        model.save(model_path)
        registry.register_model(
            model_name="test_xgboost",
            model_path=model_path,
            metadata=model.metadata,
            version="1.0.0"
        )
        metadata = registry.get_metadata("test_xgboost", "1.0.0")
        assert metadata['test_metadata'] == 'test_value'
        assert metadata['metrics']['RMSE'] == 0.1

    def test_compare_versions(self, registry, test_model, temp_model_dir):
        model = test_model()
        model_path = os.path.join(temp_model_dir, 'test_model.pkl')
        model.save(model_path)
        registry.register_model(
            model_name="test_xgboost",
            model_path=model_path,
            metadata=model.metadata,
            version="1.0.0"
        )
        registry.register_model(
            model_name="test_xgboost",
            model_path=model_path,
            metadata=model.metadata,
            version="1.1.0"
        )
        comparison = registry.compare_versions("test_xgboost", "1.0.0", "1.1.0")
        assert "v1" in comparison and "v2" in comparison

    def test_delete_version(self, registry, test_model, temp_model_dir):
        model = test_model()
        model_path = os.path.join(temp_model_dir, 'test_model.pkl')
        model.save(model_path)
        registry.register_model(
            model_name="test_xgboost",
            model_path=model_path,
            metadata=model.metadata,
            version="1.0.0"
        )
        registry.delete_version("test_xgboost", "1.0.0")
        versions = registry.list_versions("test_xgboost") if "test_xgboost" in registry.list_models() else []
        assert "1.0.0" not in versions

    def test_delete_model(self, registry, test_model, temp_model_dir):
        model = test_model()
        model_path = os.path.join(temp_model_dir, 'test_model.pkl')
        model.save(model_path)
        registry.register_model(
            model_name="test_xgboost",
            model_path=model_path,
            metadata=model.metadata,
            version="1.0.0"
        )
        registry.delete_model("test_xgboost")
        assert "test_xgboost" not in registry.list_models()

    def test_export_model(self, registry, test_model, temp_model_dir):
        model = test_model()
        model_path = os.path.join(temp_model_dir, 'test_model.pkl')
        model.save(model_path)
        registry.register_model(
            model_name="test_xgboost",
            model_path=model_path,
            metadata=model.metadata,
            version="1.0.0"
        )
        export_dir = os.path.join(temp_model_dir, 'export')
        os.makedirs(export_dir, exist_ok=True)
        exported_path = registry.export_model(
            model_name="test_xgboost",
            version="1.0.0",
            export_path=export_dir
        )
        assert os.path.exists(exported_path)

    def test_batch_metadata_registration(self, registry, test_model, temp_model_dir):
        model = test_model()
        model_path = os.path.join(temp_model_dir, 'test_model.pkl')
        model.save(model_path)
        batch_metadata = [{"batch": i} for i in range(5)]
        version = registry.register_model(
            model_name="test_xgboost",
            model_path=model_path,
            metadata=model.metadata,
            version="1.0.0",
            batch_metadata=batch_metadata
        )
        metadata = registry.get_metadata("test_xgboost", version)
        assert "batch_metadata" in metadata
        assert len(metadata["batch_metadata"]) == 5

    def test_record_batch_performance(self, registry, test_model, temp_model_dir):
        model = test_model()
        model_path = os.path.join(temp_model_dir, 'test_model.pkl')
        model.save(model_path)
        version = registry.register_model(
            model_name="test_batch_metrics",
            model_path=model_path,
            metadata=model.metadata,
            version="1.0.0"
        )
        # Simulate batch performance
        batch_perf = [{"accuracy": 0.9 + i * 0.01} for i in range(2)]
        registry.register_model(
            model_name="test_batch_metrics",
            model_path=model_path,
            metadata=model.metadata,
            version="1.1.0",
            batch_performance=batch_perf
        )
        perf = registry.compare_batch_metrics("test_batch_metrics", "1.1.0")
        assert isinstance(perf, list)
        assert len(perf) == 2

    def test_compare_batch_metrics(self, registry, test_model, temp_model_dir):
        model = test_model()
        model_path = os.path.join(temp_model_dir, 'test_model.pkl')
        model.save(model_path)
        v1 = registry.register_model(
            model_name="compare_batch",
            model_path=model_path,
            metadata=model.metadata,
            version="1.0.0",
            batch_performance=[{"score": 0.8}]
        )
        v2 = registry.register_model(
            model_name="compare_batch",
            model_path=model_path,
            metadata=model.metadata,
            version="1.1.0",
            batch_performance=[{"score": 0.9}]
        )
        perf1 = registry.compare_batch_metrics("compare_batch", v1)
        perf2 = registry.compare_batch_metrics("compare_batch", v2)
        assert perf1[0]["score"] == 0.8
        assert perf2[0]["score"] == 0.9

    def test_concurrent_registration(self, registry, test_model, tmp_path_factory):
        model_paths = [tmp_path_factory.mktemp(f"model_{i}") / "model.pkl" for i in range(3)]
        def register_worker(path):
            model = test_model()
            model.save(path)
            return registry.register_model(
                "concurrent_test",
                path,
                timeout=10.0
            )
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(register_worker, p) for p in model_paths]
            results = [f.result(timeout=15.0) for f in futures]
        assert len(set(results)) == 3
        assert len(registry.get_versions("concurrent_test")) == 3

    def test_concurrent_access(self, tmp_path):
        registry = ModelRegistry(registry_path=tmp_path)
        # Initial model
        model_path = tmp_path / "base_model.pkl"
        with open(model_path, "wb") as f:
            f.write(b"base_data")
        registry.register_model("test", str(model_path))
        def _worker(i):
            if i % 3 == 0:  # Writers
                new_path = tmp_path / f"model_{i}.pkl"
                with open(new_path, "wb") as f:
                    f.write(f"data_{i}".encode())
                registry.register_model("test", str(new_path))
            else:  # Readers
                registry.get_model("test")
                registry.list_versions("test")
        with ThreadPoolExecutor(max_workers=8) as executor:
            list(executor.map(_worker, range(50)))
        assert len(registry.list_versions("test")) > 1
        assert registry.get_model("test") is not None
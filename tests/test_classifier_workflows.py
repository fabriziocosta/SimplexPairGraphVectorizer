import sys
from pathlib import Path

import joblib
import pytest
from sklearn.pipeline import Pipeline

pytest.importorskip("pandas")
pytest.importorskip("matplotlib")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebooks" / "src"))
from enzyme_workflow.classifier import _detach_fit_callbacks


class _TrainingCallback:
    pass


class _ClassifierWithCallbacks:
    def __init__(self):
        self.callbacks = [_TrainingCallback()]


def test_detaching_training_callback_makes_pipeline_serializable(tmp_path):
    class LocalTqdmCallback:
        pass

    model = Pipeline([("classifier", _ClassifierWithCallbacks())])
    model.named_steps["classifier"].callbacks = [LocalTqdmCallback()]
    _detach_fit_callbacks(model)

    assert model.named_steps["classifier"].callbacks is None
    joblib.dump(model, tmp_path / "model.joblib")

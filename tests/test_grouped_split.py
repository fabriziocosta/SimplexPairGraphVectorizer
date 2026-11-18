import numpy as np
import pytest
import sys
from pathlib import Path

pytest.importorskip("pandas")
pytest.importorskip("matplotlib")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebooks" / "src"))
from enzyme_workflow.classifier import _grouped_train_test_split


def test_stratified_group_split_keeps_all_classes_in_training_and_test():
    labels = np.array(["1.1"] * 5 + ["2.1"] * 5)
    groups = np.array([f"g{index}" for index in range(len(labels))])

    train, test, strategy = _grouped_train_test_split(
        labels, groups, test_size=0.2, random_state=17
    )

    assert set(labels[train]) == set(labels)
    assert set(labels[test]) == set(labels)
    assert set(groups[train]).isdisjoint(groups[test])
    assert "StratifiedGroupKFold" in strategy


def test_grouped_holdout_keeps_singleton_class_in_training():
    labels = np.array(["1.1", "2.1", "2.1", "2.1", "2.1", "2.1"])
    groups = np.array(["singleton", "b1", "b2", "b3", "b4", "b5"])

    with pytest.warns(UserWarning, match="grouped holdout"):
        train, test, strategy = _grouped_train_test_split(
            labels, groups, test_size=0.2, random_state=11
        )

    assert set(labels[train]) == set(labels)
    assert "1.1" not in set(labels[test])
    assert set(groups[train]).isdisjoint(groups[test])
    assert "GroupShuffleSplit" in strategy


def test_grouped_split_fails_when_every_holdout_drops_a_training_class():
    labels = np.array(["1.1", "2.1"])
    groups = np.array(["g1", "g2"])

    with pytest.warns(UserWarning, match="grouped holdout"):
        with pytest.raises(ValueError, match="every EC class in training"):
            _grouped_train_test_split(
                labels, groups, test_size=0.5, random_state=3
            )

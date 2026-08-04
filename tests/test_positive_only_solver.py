"""Tests for reconstruction.use_positive_only_solver settings wiring."""

from src.pipelines.reconstruction import use_positive_only_solver_from_settings


def test_positive_only_solver_defaults_false():
    assert use_positive_only_solver_from_settings({}) is False
    assert use_positive_only_solver_from_settings({"reconstruction": {}}) is False
    assert use_positive_only_solver_from_settings(None) is False


def test_positive_only_solver_reads_flag():
    assert (
        use_positive_only_solver_from_settings(
            {"reconstruction": {"use_positive_only_solver": True}}
        )
        is True
    )
    assert (
        use_positive_only_solver_from_settings(
            {"reconstruction": {"use_positive_only_solver": False}}
        )
        is False
    )

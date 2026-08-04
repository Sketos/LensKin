"""Tests for phase-1 reconstruction SNR masking used when locking total flux."""
import numpy as np
import pytest

from src.pipelines.reconstruction import (
    DEFAULT_FLUX_SNR_THRESHOLD,
    apply_reconstruction_snr_mask,
    flux_snr_threshold_from_settings,
)


def test_default_flux_snr_threshold_is_half():
    assert flux_snr_threshold_from_settings({}) == DEFAULT_FLUX_SNR_THRESHOLD
    assert flux_snr_threshold_from_settings({"reconstruction": {}}) == 0.5


def test_flux_snr_threshold_can_be_disabled():
    assert flux_snr_threshold_from_settings(
        {"reconstruction": {"flux_snr_threshold": None}}
    ) is None
    assert flux_snr_threshold_from_settings(
        {"reconstruction": {"flux_snr_threshold": False}}
    ) is None
    assert flux_snr_threshold_from_settings(
        {"reconstruction": {"flux_snr_threshold": 0}}
    ) is None
    assert flux_snr_threshold_from_settings(
        {"reconstruction": {"flux_snr_threshold": -1}}
    ) is None


def test_flux_snr_threshold_reads_custom_value():
    assert flux_snr_threshold_from_settings(
        {"reconstruction": {"flux_snr_threshold": 2.0}}
    ) == 2.0


def test_apply_reconstruction_snr_mask_keeps_high_snr_pixels():
    sb = np.array([[0.0, 0.1], [0.5, 1.0]])
    noise = np.array([[0.1, 0.1], [0.1, 0.1]])
    masked = apply_reconstruction_snr_mask(sb, noise, snr_threshold=0.5)
    # SNR = [0, 1, 5, 10] → keep last three with thr=0.5? 0.1/0.1=1, etc.
    # 0.0/0.1 = 0 < 0.5 → zeroed; others kept
    np.testing.assert_allclose(masked, np.array([[0.0, 0.1], [0.5, 1.0]]))


def test_apply_reconstruction_snr_mask_removes_negatives_for_positive_threshold():
    sb = np.array([[-1.0, 0.2], [0.05, 1.0]])
    noise = np.full_like(sb, 0.1)
    masked = apply_reconstruction_snr_mask(sb, noise, snr_threshold=1.0)
    # SNR: -10, 2, 0.5, 10 → keep 2 and 10
    np.testing.assert_allclose(masked, np.array([[0.0, 0.2], [0.0, 1.0]]))


def test_apply_reconstruction_snr_mask_none_is_noop():
    sb = np.array([[-1.0, 2.0]])
    noise = np.array([[0.1, 0.1]])
    out = apply_reconstruction_snr_mask(sb, noise, snr_threshold=None)
    np.testing.assert_allclose(out, sb)

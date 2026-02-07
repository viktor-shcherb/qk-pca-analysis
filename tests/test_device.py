"""Tests for qk_pca.device."""

from __future__ import annotations

from unittest.mock import patch

import torch

from qk_pca.device import get_device


class TestGetDevice:
    def test_explicit_cpu(self):
        assert get_device("cpu") == torch.device("cpu")

    def test_explicit_cuda(self):
        assert get_device("cuda") == torch.device("cuda")

    def test_explicit_mps(self):
        assert get_device("mps") == torch.device("mps")

    def test_auto_picks_cuda_when_available(self):
        with patch("torch.cuda.is_available", return_value=True):
            assert get_device("auto") == torch.device("cuda")

    def test_auto_picks_mps_when_no_cuda(self):
        with patch("torch.cuda.is_available", return_value=False), \
             patch("torch.backends.mps.is_available", return_value=True):
            assert get_device("auto") == torch.device("mps")

    def test_auto_falls_back_to_cpu(self):
        with patch("torch.cuda.is_available", return_value=False), \
             patch("torch.backends.mps.is_available", return_value=False):
            assert get_device("auto") == torch.device("cpu")

"""Offline probes must use real current observations and check readout effects."""
from types import SimpleNamespace

import pytest
import torch

from tools.smoke_rgb_inference import ensure_probe_observation, validate_readout_probe


def test_probe_installs_initial_real_observation_once_and_reinstalls_after_reset():
    encoded = torch.randn(1, 4, 1, 2, 3, requires_grad=True)
    observations = [object(), object(), object()]
    calls = []

    def encode(value):
        calls.append(value)
        return encoded

    server = SimpleNamespace(transformer=SimpleNamespace(uses_rgb_readout=True),
                             readout_observation_latent=None, _encode_initial_obs=encode)
    result = ensure_probe_observation(server, observations)
    assert calls == [{"obs": observations[:1]}]
    torch.testing.assert_close(result, encoded, atol=0, rtol=0)
    assert not result.requires_grad and result.data_ptr() != encoded.data_ptr()
    assert ensure_probe_observation(server, observations) is result
    assert len(calls) == 1
    # A new actual observation must be preserved; probes cannot overwrite it
    # with the initial frame once the execution window has progressed.
    advanced = torch.randn_like(encoded)
    server.readout_observation_latent = advanced
    assert ensure_probe_observation(server, observations) is advanced
    assert len(calls) == 1
    server.readout_observation_latent = None
    ensure_probe_observation(server, observations)
    assert len(calls) == 2


def test_legacy_probe_does_not_require_or_encode_a_readout_observation():
    server = SimpleNamespace(transformer=SimpleNamespace(uses_rgb_readout=False))
    assert ensure_probe_observation(server, [object()]) is None


@pytest.mark.parametrize("mode", ["static", "state", "global_local"])
def test_readout_control_accepts_expected_state_dependence(mode):
    initial = torch.randn(1, 8, 12)
    changed = initial.clone()
    if mode == "state":
        changed += 1
    elif mode == "global_local":
        changed[:, 2:] += 1
    result = validate_readout_probe(mode, initial, changed, 2)
    assert result["shape"] == [1, 8, 12]
    assert result["max_abs"] == pytest.approx(0 if mode == "static" else 1)


@pytest.mark.parametrize("mode", ["state", "global_local"])
def test_readout_control_rejects_a_disconnected_state_path(mode):
    initial = torch.randn(1, 8, 12)
    with pytest.raises(AssertionError, match="ignored a state intervention"):
        validate_readout_probe(mode, initial, initial.clone(), 2)


@pytest.mark.parametrize("mode", ["static", "global_local"])
def test_readout_control_rejects_state_leak_into_invariant_queries(mode):
    initial = torch.randn(1, 8, 12)
    changed = initial + 1
    with pytest.raises(AssertionError):
        validate_readout_probe(mode, initial, changed, 2)


def test_readout_control_rejects_nonfinite_or_shape_mismatch():
    initial = torch.randn(1, 8, 12)
    with pytest.raises(AssertionError, match="shapes differ"):
        validate_readout_probe("state", initial, initial[:, :2], 2)
    changed = initial.clone()
    changed[0, 0, 0] = torch.nan
    with pytest.raises(AssertionError, match="Nonfinite"):
        validate_readout_probe("state", initial, changed, 2)

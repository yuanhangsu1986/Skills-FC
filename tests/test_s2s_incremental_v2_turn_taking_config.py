import pytest

from recipes.multimodal.server.backends.s2s_incremental_backend_v2 import (
    S2SIncrementalBackendV2,
    S2SIncrementalV2Config,
)


@pytest.mark.parametrize("source", ["asr_head", "rnnt", "both"])
def test_turn_taking_config_is_forwarded_to_drirf(source):
    config = S2SIncrementalV2Config.from_dict(
        {
            "model_path": "/tmp/test-model",
            "turn_taking_source": source,
            "rnnt_eou_frames": 17,
            "rnnt_bou_frames": 5,
            "rnnt_min_speech_frames": 4,
            "rnnt_max_symbols": 11,
        }
    )

    assert "turn_taking_source" not in config.extra_config

    wrapper_config = S2SIncrementalBackendV2(config)._build_wrapper_config()
    assert wrapper_config.turn_taking_source == source
    assert wrapper_config.rnnt_eou_frames == 17
    assert wrapper_config.rnnt_bou_frames == 5
    assert wrapper_config.rnnt_min_speech_frames == 4
    assert wrapper_config.rnnt_max_symbols == 11


def test_turn_taking_defaults_to_legacy_asr_head():
    config = S2SIncrementalV2Config(model_path="/tmp/test-model")
    wrapper_config = S2SIncrementalBackendV2(config)._build_wrapper_config()

    assert wrapper_config.turn_taking_source == "asr_head"

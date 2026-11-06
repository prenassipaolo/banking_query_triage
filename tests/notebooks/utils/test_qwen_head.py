"""Check head training and frozen feature extraction without downloading Qwen."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

# The main serving/test environment deliberately does not depend on PyTorch.
torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
from notebooks.utils import qwen_head


@pytest.mark.parametrize("interrupted", [False, True])
def test_features_pool_resume_and_reuse_cache(tmp_path, monkeypatch, interrupted):
    calls = 0

    class Tokenizer:
        pad_token_id = 0

        def __call__(self, texts, **kwargs):
            rows = {
                "a": [4, 5, 0],
                "b": [6, 7, 8],
                "c": [14, 15, 0],
                "d": [16, 17, 18],
            }
            ids = torch.tensor([rows[text] for text in texts])
            return {
                "input_ids": ids,
                "attention_mask": ids.ne(0).long(),
            }

    class Backbone(torch.nn.Module):
        def forward(self, input_ids, attention_mask, **kwargs):
            nonlocal calls
            calls += 1
            if interrupted and calls == 2:
                raise RuntimeError("simulated interruption")
            assert not self.training
            assert not torch.is_grad_enabled()
            return SimpleNamespace(last_hidden_state=input_ids.unsqueeze(-1).float())

    monkeypatch.setattr(
        qwen_head.AutoTokenizer, "from_pretrained", lambda *a, **k: Tokenizer()
    )
    monkeypatch.setattr(
        qwen_head.AutoModel, "from_pretrained", lambda *a, **k: Backbone()
    )
    config = {
        "model_id": "fake",
        "revision": "fixed",
        "max_length": 3,
        "feature_batch_size": 2,
    }
    texts = ["a", "b", "c", "d"]
    if interrupted:
        with pytest.raises(RuntimeError, match="simulated interruption"):
            qwen_head.extract_features(texts, config, tmp_path)
    features = qwen_head.extract_features(texts, config, tmp_path)
    assert features.flatten().tolist() == [5, 8, 15, 18]
    assert calls == (3 if interrupted else 2)
    monkeypatch.setattr(
        qwen_head.AutoModel,
        "from_pretrained",
        lambda *a, **k: pytest.fail("cache missed"),
    )
    assert torch.equal(features, qwen_head.extract_features(texts, config, tmp_path))


def test_early_stopping_restores_best_validation_checkpoint():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    x = torch.randn(24, 8)
    y = torch.arange(24) % 3
    config = {
        "seed": 42,
        "classes": 3,
        "dropout": 0.2,
        "learning_rate": 0.01,
        "weight_decay": 0.01,
        "batch_size": 8,
        "max_epochs": 10,
        "patience": 2,
        "min_delta": 100.0,
    }
    model, history, snapshots, best = qwen_head.fit_head(
        x, y, np.arange(18), np.arange(18, 24), 4, config
    )
    assert best == 1
    assert len(history) == 3
    assert snapshots == []
    assert qwen_head.score(model, x[18:], y[18:])[0] == pytest.approx(
        history[0]["validation_loss"]
    )
    model, history, snapshots, best = qwen_head.fit_head(
        x, y, np.arange(24), None, 4, config, epochs=2
    )
    assert best == len(history) == len(snapshots) == 2
    assert all("validation_loss" not in row for row in history)
    assert all(
        torch.equal(value, snapshots[-1][key])
        for key, value in model.state_dict().items()
    )


def test_sampling_keeps_normalized_duplicates_together():
    frame = pd.DataFrame(
        {
            "text": ["Hi", " HI ", "Other", "Bye", "Next"],
            "category": ["a", "a", "a", "b", "b"],
        }
    )
    sampled = qwen_head.sample_frame(frame, 2, 42)
    assert sampled.groupby("category").size().tolist() == [2, 2]
    assert sampled.group.is_unique

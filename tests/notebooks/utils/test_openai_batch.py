"""Exercise asynchronous row mapping and failures without uploading or billing."""

import json
from types import SimpleNamespace

import pandas as pd
import pytest

from notebooks.utils.model_comparison import ComparisonConfig
from notebooks.utils.openai_batch import collect_batch, prepare_batch, submit_batch


@pytest.mark.parametrize("malformed", [False, True])
def test_batch_preparation_and_unordered_collection_preserve_failed_rows(
    tmp_path, malformed
):
    (tmp_path / "config").mkdir()
    (tmp_path / "config/serving.toml").write_text('model_dir="unused"\n')
    (tmp_path / "labels.json").write_text('["card", "cash"]')
    frame = pd.DataFrame(
        {"text": ["card query", "cash query"], "category": ["card", "cash"]}
    )
    frame.to_csv(tmp_path / "train.csv", index=False)
    frame.to_csv(tmp_path / "test.csv", index=False)
    config = ComparisonConfig(
        tmp_path,
        "http://unused",
        "train.csv",
        "test.csv",
        "labels.json",
        "out",
        ["openai"],
        42,
        1,
        1,
        10,
    )
    directory = prepare_batch(config)
    manifest = json.loads((directory / "manifest.json").read_text())
    requests = [
        json.loads(line)
        for line in (directory / "input.jsonl").read_text().splitlines()
    ]
    assert len(requests) == 4
    assert all(
        request["body"]["reasoning"] == {"effort": "none"} for request in requests
    )
    assert all(
        set(item) == {"message_index", "text"}
        for request in requests
        for item in json.loads(request["body"]["input"])
    )
    assert prepare_batch(config) == directory
    # Terminal job with reversed output order and one failed request.
    output = []
    for request in reversed(requests[1:]):
        chunk = manifest["chunks"][request["custom_id"]]
        category = manifest["truth"][chunk["split"]][str(chunk["row_ids"][0])]
        body = {
            "status": "completed",
            "model": "resolved-model",
            "usage": {"input_tokens": 10, "output_tokens": 5},
            "output": [
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "text": json.dumps(
                                {
                                    "predictions": [
                                        {
                                            "message_index": 0,
                                            "category": category,
                                            "confidence": 0.9,
                                        }
                                    ]
                                }
                            ),
                        }
                    ],
                }
            ],
        }
        if malformed and not output:
            body["output"] = []
        output.append(
            {
                "custom_id": request["custom_id"],
                "response": {"status_code": 200, "body": body},
            }
        )
    job = SimpleNamespace(
        id="batch-1",
        status="completed",
        output_file_id="output",
        error_file_id=None,
        created_at=100,
        completed_at=110,
        expired_at=None,
        cancelled_at=None,
        failed_at=None,
        model_dump_json=lambda **kwargs: '{"id":"batch-1"}',
    )
    client = SimpleNamespace(
        files=SimpleNamespace(
            create=lambda **kwargs: SimpleNamespace(id="file-1"),
            content=lambda file_id: SimpleNamespace(
                text="\n".join(json.dumps(line) for line in output)
            ),
        ),
        batches=SimpleNamespace(
            create=lambda **kwargs: job, retrieve=lambda job_id: job
        ),
    )
    assert submit_batch(directory, client)["id"] == "batch-1"
    with pytest.raises(FileExistsError, match="submitted"):
        submit_batch(directory, client)
    summary = collect_batch(directory, client)
    assert summary.attempted_rows.sum() == 4
    assert summary.valid_rows.sum() == (2 if malformed else 3)
    usage = json.loads((directory / "usage.json").read_text())
    assert len(usage) == 4
    assert sum(item["usage"] is not None for item in usage) == 3
    assert summary.batch_p50_seconds.isna().all()
    assert summary.amortized_ms_per_message.isna().all()
    assert summary.job_turnaround_seconds.tolist() == [10, 10]
    pd.testing.assert_frame_equal(summary, collect_batch(directory, client))
    (directory / "input.jsonl").write_text("changed input\n")
    with pytest.raises(ValueError, match="input changed"):
        submit_batch(
            directory, SimpleNamespace()
        )  # Must fail before accessing the client.
    with pytest.raises(ValueError, match="input changed"):
        collect_batch(directory, SimpleNamespace())

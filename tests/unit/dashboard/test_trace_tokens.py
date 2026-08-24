from prime_rl.dashboard import server


class FakeTokenizer:
    def decode(self, ids: list[int], *, skip_special_tokens: bool) -> str:
        assert skip_special_tokens is False
        pieces = {1: "<bos>", 2: "system", 3: "<tools>", 4: "assistant", 5: "<eos>"}
        return "".join(pieces[token_id] for token_id in ids)


def branching_trace() -> dict:
    return {
        "nodes": [
            {"token_ids": [1, 2]},
            {"parent": 0, "token_ids": [3]},
            {"parent": 1, "token_ids": [4, 5]},
            {"parent": 0, "token_ids": [5]},
        ]
    }


def test_rendered_token_text_decodes_full_recorded_branches(monkeypatch) -> None:
    monkeypatch.setattr(server, "get_tokenizer", lambda model: FakeTokenizer())

    rendered = server.rendered_token_text(branching_trace(), "synthetic/renderer")

    assert rendered["status"] == "ok"
    assert rendered["model"] == "synthetic/renderer"
    assert rendered["paths"] == [
        {"nodes": [0, 1, 2], "token_count": 5, "text": "<bos>system<tools>assistant<eos>"},
        {"nodes": [0, 3], "token_count": 3, "text": "<bos>system<eos>"},
    ]
    assert rendered["all_nodes"] == {
        "nodes": [0, 1, 2, 3],
        "token_count": 6,
        "text": "<bos>system<tools>assistant<eos><eos>",
    }


def test_rendered_token_text_reports_missing_token_ids() -> None:
    rendered = server.rendered_token_text({"nodes": [{"message": {"role": "user"}}]}, "synthetic/renderer")

    assert rendered == {"status": "missing_token_ids", "model": "synthetic/renderer", "paths": []}


def test_rendered_token_text_reports_missing_model() -> None:
    rendered = server.rendered_token_text({"nodes": [{"token_ids": [1]}]}, None)

    assert rendered == {"status": "missing_model", "model": None, "paths": []}


def test_rendered_token_text_reports_unavailable_tokenizer(monkeypatch) -> None:
    monkeypatch.setattr(server, "get_tokenizer", lambda model: None)

    rendered = server.rendered_token_text({"nodes": [{"token_ids": [1]}]}, "synthetic/missing")

    assert rendered == {"status": "tokenizer_unavailable", "model": "synthetic/missing", "paths": []}

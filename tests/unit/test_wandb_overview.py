from prime_rl.monitors.wandb import overview


def test_list_views_uses_workspace_graphql_transport(monkeypatch) -> None:
    api = object()
    response = {
        "project": {
            "allViews": {
                "edges": [
                    {"node": {"displayName": "overview", "name": "nw-overview-v"}},
                    {"node": None},
                ]
            }
        }
    }

    monkeypatch.setattr(overview.wandb, "Api", lambda: api)

    def execute(received_api: object, query: str, variables: dict[str, str]) -> dict:
        assert received_api is api
        assert isinstance(query, str)
        assert "query Views" in query
        assert variables == {"entity": "prime", "project": "rl"}
        return response

    monkeypatch.setattr(overview, "execute_graphql", execute)

    assert overview.list_views("prime", "rl") == [("overview", "nw-overview-v")]

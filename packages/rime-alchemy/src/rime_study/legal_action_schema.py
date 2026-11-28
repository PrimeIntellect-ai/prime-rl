"""Visible-state action grammar for immutable actor inference."""


def visible_ids(observation, field):
    rows = observation[field]
    ids = [row["id"] for row in rows]
    if any(type(index) is not int or index < 0 for index in ids) or len(ids) != len(set(ids)):
        raise ValueError(f"Invalid visible {field} IDs")
    return sorted(ids)


def action_schema(observation):
    """Constrain action shape and current IDs; provide no reward or hidden facts."""
    stones = visible_ids(observation, "stones")
    potions = visible_ids(observation, "potions")

    def branch(operation, **fields):
        properties = {"op": {"type": "string", "enum": [operation]}, **fields}
        return {"type": "object", "properties": properties,
                "required": list(properties), "additionalProperties": False}

    branches = [branch("end_trial")]
    if stones:
        stone = {"type": "integer", "enum": stones}
        branches.append(branch("cauldron", stone=stone))
        if potions:
            branches.append(branch("potion", stone=stone,
                                   potion={"type": "integer", "enum": potions}))
    return {"anyOf": branches}

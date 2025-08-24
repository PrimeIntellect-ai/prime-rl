"""
This test is used to ensure that the moe venv is up to date with the main venv.
This split is currently needed because torchtitan moe requires torch 2.8 but vllm requires torch 2.7.

If this test fails, please sync the changes from pyproject.toml to deps/moe/pyproject.toml.
Copy it from deps/moe/pyproject.toml to pyproject.toml.
Then run `uv sync && uv sync --all-extras` to update the venv and lock.
Then copy the updated uv.lock to deps/moe/uv.lock.
Commit the changes in deps/moe/*.
"""

import tomllib

MAIN_PYPROJECT = "pyproject.toml"
MOE_PYPROJECT = "deps/moe/pyproject.toml"


def test_moe_venv():
    with open(MAIN_PYPROJECT, "rb") as f:
        main = tomllib.load(f)
    with open(MOE_PYPROJECT, "rb") as f:
        moe = tomllib.load(f)

    def apply_moe_changes(pyproject_dict, moe_dict):
        for i in range(len(pyproject_dict["project"]["dependencies"])):
            if "torch" in pyproject_dict["project"]["dependencies"][i]:
                pyproject_dict["project"]["dependencies"][i] = "torch>=2.8.0"
        pyproject_dict["project"]["dependencies"].extend(["torchtitan", "blobfile>=3.0.0"])
        pyproject_dict["project"]["optional-dependencies"]["flash-attn"] = moe_dict["project"]["optional-dependencies"][
            "flash-attn"
        ]

        pyproject_dict["tool"]["uv"]["sources"]["torchtitan"] = moe_dict["tool"]["uv"]["sources"]["torchtitan"]
        pyproject_dict["tool"]["uv"]["override-dependencies"] = moe_dict["tool"]["uv"]["override-dependencies"]

    apply_moe_changes(main, moe)

    for k, v in main.items():
        assert k in moe, f"{k} not in moe venv"
        assert v == moe[k], f"'{k}' key not equal. Please sync the changes on {MAIN_PYPROJECT} to {MOE_PYPROJECT}."

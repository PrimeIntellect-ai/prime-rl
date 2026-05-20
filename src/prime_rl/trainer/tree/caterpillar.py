from prime_rl.trainer.tree.tree import Tree, TreeNode


def build_caterpillar(
    turns: list[tuple[list[int], list[int], list[int]]],
    train_response: bool = True,
    train_think: bool = True,
) -> Tree:
    if len(turns) == 0:
        raise ValueError("Caterpillar tree requires at least one turn")

    nodes: list[TreeNode] = []
    trunk_parent = -1
    for turn_idx, (user_ids, think_ids, response_ids) in enumerate(turns):
        user_idx = len(nodes)
        nodes.append(TreeNode(parent=trunk_parent, token_ids=user_ids, loss_mask=[False] * len(user_ids)))

        think_idx = len(nodes)
        nodes.append(TreeNode(parent=user_idx, token_ids=think_ids, loss_mask=[train_think] * len(think_ids)))

        response_idx = len(nodes)
        nodes.append(TreeNode(parent=user_idx, token_ids=response_ids, loss_mask=[train_response] * len(response_ids)))

        if nodes[think_idx].parent != nodes[response_idx].parent:
            raise AssertionError(f"Turn {turn_idx} think and response nodes must share the user parent")

        trunk_parent = response_idx

    return Tree(nodes)

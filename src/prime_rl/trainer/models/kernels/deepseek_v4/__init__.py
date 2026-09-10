IGNORE_SLOT = -1
"""A slot is one key a query may read. `IGNORE_SLOT` stands in a slot where the query has no key
at all.

Every query is given the same maximum number of slots, so a query with fewer keys to read leaves
the rest empty, and an empty slot contributes nothing to its attention.
"""

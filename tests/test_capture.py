from __future__ import annotations

from collections import Counter

from rps.cli.capture import generate_prompts


def test_prompts_are_balanced_without_triples() -> None:
    prompts = generate_prompts(20, seed=42)
    assert Counter(prompts) == {0: 20, 1: 20, 2: 20}
    assert all(
        not (prompts[index] == prompts[index - 1] == prompts[index - 2])
        for index in range(2, len(prompts))
    )

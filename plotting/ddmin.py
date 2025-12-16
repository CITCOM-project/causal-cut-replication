"""
Delta debugging algorithm.

Implementation taken from The Debugging Book, Andreas Zeller.
https://www.debuggingbook.org/beta/DeltaDebugger.html
Accessed 2025-12-16
"""

from typing import Callable, Sequence, Any

PASS = False
FAIL = True


def ddmin(test: Callable, inp: Sequence[Any], *test_args: Any) -> Sequence:
    """
    Reduce `inp` to a 1-minimal failing subset, using the outcome
    of `test(inp, *test_args)`, which should be `PASS`, `FAIL`, or `UNRESOLVED`.
    """
    assert test(inp, *test_args) != PASS
    executions = 1

    n = 2  # Initial granularity
    while len(inp) >= 2:
        start: int = 0  # Where to start the next subset
        subset_length: int = int(len(inp) / n)
        some_complement_is_failing: bool = False

        while start < len(inp):
            # Cut out inp[start:(start + subset_length)]
            complement: Sequence[Any] = inp[:start] + inp[start + subset_length :]  # type: ignore

            executions += 1
            if test(complement, *test_args) == FAIL:
                # Continue with reduced input
                inp = complement
                n = max(n - 1, 2)
                some_complement_is_failing = True
                break

            # Continue with next subset
            start += subset_length

        if not some_complement_is_failing:
            # Increase granularity
            if n == len(inp):
                break
            n = min(n * 2, len(inp))

    return inp, executions

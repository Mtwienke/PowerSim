"""Bounded, ordered process parallelism for independent pandapower cases."""
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from functools import partial
from itertools import islice
from multiprocessing import get_context
import os

THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS")


def validate_workers(count):
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("workers must be a positive integer")


def _batch(function, items):
    return [function(item) for item in items]


@contextmanager
def ordered_pool(workers, initializer, initargs=()):
    """Yield an ordered map. Only a few batches are in flight at a time.

    CLI/process entry points only: set child thread limits before spawn (and
    restore the caller environment afterward). Do not fork a live ML runtime.
    """
    validate_workers(workers)
    if workers == 1:
        initializer(*initargs)
        yield map
        return

    original = {name: os.environ.get(name) for name in THREAD_ENV}
    os.environ.update(dict.fromkeys(THREAD_ENV, "1"))
    try:
        with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn"),
                                 initializer=initializer, initargs=initargs) as executor:
            def ordered_map(function, items):
                source = iter(items)
                pending = deque()

                def submit():
                    group = list(islice(source, 4))
                    if group:
                        pending.append(executor.submit(partial(_batch, function), group))

                for _ in range(2 * workers):
                    submit()
                while pending:
                    group = pending.popleft().result()
                    submit()
                    yield from group

            yield ordered_map
    finally:
        for name, value in original.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

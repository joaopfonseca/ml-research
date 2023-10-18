import pytest
import os
import time
import contextlib
import io
from mlresearch.utils._parallelize import _get_n_jobs, parallel_loop


def example_function(a):
    time.sleep(0.5)
    return None


def test_get_n_jobs():
    max_jobs = os.cpu_count()
    other_n_jobs = 2 if max_jobs >= 2 else 1

    assert _get_n_jobs(None) == 1
    assert _get_n_jobs(other_n_jobs) == other_n_jobs
    assert _get_n_jobs(-1) == max_jobs

    with pytest.raises(RuntimeError):
        _get_n_jobs(max_jobs + 1)


def test_parallel_loop():
    # Get number of jobs to set iterable range
    n_jobs = os.cpu_count()

    # Check if parallelization is happening
    start = time.time()
    parallel_loop(example_function, range(n_jobs), n_jobs=-1)
    exec_time = time.time() - start

    exp_single_job = n_jobs * 0.5 if n_jobs != 1 else 1
    assert exec_time < exp_single_job


def test_progress_bar():
    # Get number of jobs to set iterable range
    n_jobs = os.cpu_count()
    description = "Test descr."

    # Check if parallelization is happening
    f = io.StringIO()
    with contextlib.redirect_stderr(f):
        parallel_loop(
            function=example_function,
            iterable=range(n_jobs),
            n_jobs=-1,
            progress_bar=True,
            description=description,
        )

    output = f.getvalue().splitlines()[1:]
    assert all([out_.startswith(description) for out_ in output])

import os
from joblib import Parallel, delayed
from ._utils import _optional_import


def _get_n_jobs(n_jobs):
    """Assign number of jobs to be assigned in parallel."""
    max_jobs = os.cpu_count()
    n_jobs = 1 if n_jobs is None else int(n_jobs)
    if n_jobs > max_jobs:
        raise RuntimeError("Cannot assign more jobs than the number of CPUs.")
    elif n_jobs == -1:
        return max_jobs
    else:
        return n_jobs


def parallel_loop(
    function, iterable, n_jobs=None, progress_bar=False, description=None
):
    """
    Parallelize a loop and optionally add a progress bar.

    .. warning::
        The progress bar tracks job starts, not completions.

    Parameters
    ----------
    function : function
        The function to which the elements in the iterable will passed to. Must have a
        single parameter.

    iterable : iterable
        Object to be looped over.

    n_jobs : int, default=None
        Number of jobs to run in parallel. None means 1 unless in a
        joblib.parallel_backend context. -1 means using all processors.

    Returns
    -------
    output : list
        The list with the results produced using ``function`` across ``iterable``.
    """
    if progress_bar:
        track = _optional_import("rich.progress").track
        iterable = track(iterable, description=description)

    n_jobs = _get_n_jobs(n_jobs)
    return Parallel(n_jobs=n_jobs)(delayed(function)(i) for i in iterable)

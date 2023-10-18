import os
import contextlib
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


@contextlib.contextmanager
def _tqdm_joblib(tqdm_object):
    """
    Context manager to patch joblib to report into tqdm progress bar given as argument.
    """

    def tqdm_print_progress(self):
        if self.n_completed_tasks > tqdm_object.n:
            n_completed = self.n_completed_tasks - tqdm_object.n
            tqdm_object.update(n=n_completed)

    original_print_progress = Parallel.print_progress
    Parallel.print_progress = tqdm_print_progress

    try:
        yield tqdm_object
    finally:
        Parallel.print_progress = original_print_progress
        tqdm_object.close()


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
    n_jobs = _get_n_jobs(n_jobs)

    if progress_bar:
        tqdm = _optional_import("tqdm.auto").tqdm

        with _tqdm_joblib(tqdm(desc=description, total=len(iterable))) as progress_bar:
            return Parallel(n_jobs=n_jobs)(delayed(function)(i) for i in iterable)

    else:
        return Parallel(n_jobs=n_jobs)(delayed(function)(i) for i in iterable)

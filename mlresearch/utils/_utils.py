from os.path import join, dirname, abspath
import types


def _optional_import(module: str) -> types.ModuleType:
    """
    Import an optional dependency.

    Parameters
    ----------
    module : str
        The identifier for the backend. Either an entrypoint item registered
        with importlib.metadata, "matplotlib", or a module name.

    Returns
    -------
    types.ModuleType
        The imported backend.
    """
    # This function was adapted from the _load_backend function from the pandas.plotting
    # source code.
    import importlib

    # Attempt an import of an optional dependency here and raise an ImportError if
    # needed.
    try:
        module_ = importlib.import_module(module)
    except ImportError:
        mod = module.split(".")[0]
        raise ImportError(f"{mod} is required to use this functionality.") from None

    return module_


def generate_paths(filepath):
    """
    Generate data, results and analysis paths.
    """
    prefix_path = join(dirname(abspath(filepath)), "..")
    paths = [join(prefix_path, name) for name in ("data", "results", "analysis")]
    return paths

from os.path import join, dirname, abspath


def generate_paths(filepath):
    """
    Generate data, results and analysis paths.
    """
    prefix_path = join(dirname(abspath(filepath)), "..")
    paths = [join(prefix_path, name) for name in ("data", "results", "analysis")]
    return paths

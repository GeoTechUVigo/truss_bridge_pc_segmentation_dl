import pathlib

def check_path(path):
    if path is not None:
        path = pathlib.Path(path)

        if not path.is_dir(): raise ValueError("folder {} does not exist.".format(str(path)))

    return path
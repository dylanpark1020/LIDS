from pathlib import Path


def get_text(file_path, encoding="utf-8"):
    """
    Read a text file and return its contents as a string.

    Parameters
    ----------
    file_path : str or pathlib.Path
        Path to the text file.
    encoding : str, optional
        Text encoding. Defaults to UTF-8.
    """

    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(f"The file '{path}' was not found.")

    return path.read_text(encoding=encoding)

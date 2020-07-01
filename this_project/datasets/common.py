import os
import requests


_BASE_FOLDER = "/tmp/datasets"


def get_datasets_dir(subdir):
    return _BASE_FOLDER + "/" + subdir


def created_datasets_dir(subdir=None):
    if not os.path.exists(_BASE_FOLDER):
        os.mkdir(_BASE_FOLDER)

    if subdir:
        absolute_subfolder = get_datasets_dir(subdir)
        if not os.path.exists(absolute_subfolder):
            os.mkdir(absolute_subfolder)


def fetch_asset(url, subdir):
    filename = url.split("/")[-1]

    filepath = get_datasets_dir(subdir) + "/" + filename
    if not os.path.exists(filepath):
        r = requests.get(url, allow_redirects=True)
        open(filepath, "wb").write(r.content)

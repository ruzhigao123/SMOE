from os import environ
from os.path import dirname, abspath, expanduser
from socket import gethostname


def get_root_path():
    """返回 multimodal_moe/ 根目录。"""
    return dirname(dirname(abspath(__file__)))


def get_user():
    try:
        return expanduser("~").split("/")[-1]
    except Exception:
        return "user"


def get_host():
    host = environ.get("HOSTNAME")
    return host if host is not None else gethostname()

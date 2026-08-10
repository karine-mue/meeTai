import copy
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from life import load  # noqa: E402


@pytest.fixture(scope="session")
def record():
    return load()


@pytest.fixture
def clone():
    """壊してよいコピーを作る。"""
    def _clone():
        return copy.deepcopy(load())
    return _clone


def has(errs, prefix):
    """errs の中に、指定の検査ID で始まるメッセージがあるか。"""
    return any(e.startswith(prefix) for e in errs)

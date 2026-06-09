"""Smoke tests: the package imports and version is exposed."""

import govgridmarl


def test_version():
    assert isinstance(govgridmarl.__version__, str)
    assert govgridmarl.__version__.count(".") == 2


def test_seeding_returns_seed():
    from govgridmarl.utils import set_seed

    assert set_seed(123) == 123

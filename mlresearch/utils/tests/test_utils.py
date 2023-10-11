import pytest
from .._utils import _optional_import


def test_optional_import():
    """Test if function ``_optional_import`` is working as expected."""
    np = _optional_import("numpy")
    assert np.__name__ == "numpy"

    exp_err_msg = "this_package_does_not_exist is required to use this functionality."
    with pytest.raises(ImportError) as err_msg:
        _optional_import("this_package_does_not_exist")

    assert err_msg.value.args[0] == exp_err_msg

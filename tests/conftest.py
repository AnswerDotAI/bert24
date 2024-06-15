import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-triton",
        action="store_true",
        help="run the tests only in case of that command line (marked with marker @no_cmd)",
    )


def pytest_runtest_setup(item):
    if "triton" in item.keywords and not item.config.getoption("--run-triton"):
        pytest.skip("pass --run_triton option to run this test")

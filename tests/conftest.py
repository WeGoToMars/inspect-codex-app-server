"""Register artifact retention before pytest parses command-line options."""

import pytest


def pytest_addoption(parser):
    parser.addoption("--keep-failed-artifacts", metavar="DIR",
                     help="Retain owned workspaces for failed setup/call checks.")



@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    report = (yield).get_result()
    if report.failed:
        item.test_failed = True


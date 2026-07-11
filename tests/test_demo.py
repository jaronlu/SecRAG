import httpx

from scripts.demo import build_client_timeout


def test_build_client_timeout_separates_connect_and_read_timeout():
    timeout = build_client_timeout(150.0)

    assert isinstance(timeout, httpx.Timeout)
    assert timeout.connect == 5.0
    assert timeout.read == 150.0

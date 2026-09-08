"""One trusted HTTPS request, supervised by the caller; no retries or redirects."""

import http.client
import json
import os
import subprocess
import sys
import time
from pathlib import Path

MAX_RESPONSE_BYTES = 4 * 2**20


def supervise(command, directory, deadline):
    """Bound the local transport process; remote cancellation is not implied."""
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("request deadline already reached")
    with (directory / "transport.log").open("wb") as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        try:
            child.wait(timeout=max(0, deadline - time.monotonic()))
        except BaseException:
            child.kill()
            child.wait()
            raise
        if time.monotonic() >= deadline:
            raise TimeoutError("request completed after the deadline")
        if child.returncode != 0:
            raise RuntimeError(f"transport exited {child.returncode}; see retained transport files")


def decode(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate response key")
            result[key] = value
        return result

    def nonfinite(value):
        raise ValueError("nonfinite response number")

    return json.loads(raw, object_pairs_hook=unique, parse_constant=nonfinite)


def send(request, directory, deadline):
    # The controller has already persisted these exact bytes before dispatch.
    raw = (directory / "request.json").read_bytes()
    if decode(raw) != request:
        raise ValueError("persisted request differs from the reserved request")
    supervise(
        [
            sys.executable,
            "-I",
            "-B",
            str(Path(__file__).resolve()),
            str(directory.resolve()),
            str(max(0.001, deadline - time.monotonic())),
        ],
        directory,
        deadline,
    )
    result = decode((directory / "http.json").read_bytes())
    if result["status"] != 200:
        raise RuntimeError(f"Responses HTTP status {result['status']}; no retry")
    if result.get("body_complete") is not True:
        raise RuntimeError("incomplete response body; no retry")
    return decode((directory / "response.bin").read_bytes())


def exchange(directory, socket_timeout):
    """Only this trusted child reads the key; headers never enter the archive."""
    connection = None
    try:
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("API credential unavailable")
        connection = http.client.HTTPSConnection("api.openai.com", timeout=socket_timeout)
        connection.request(
            "POST",
            "/v1/responses",
            body=(directory / "request.json").read_bytes(),
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        )
        response = connection.getresponse()
        metadata = dict(
            status=response.status,
            request_id=response.getheader("x-request-id"),
            body_complete=False,
        )
        (directory / "http.json").write_text(json.dumps(metadata) + "\n")
        size = 0
        # Unbuffered writes keep the received prefix if supervision kills this child.
        with (directory / "response.bin").open("wb", buffering=0) as raw:
            while block := response.read1(min(65536, MAX_RESPONSE_BYTES + 1 - size)):
                raw.write(block)
                size += len(block)
                if size > MAX_RESPONSE_BYTES:
                    raise ValueError("response exceeded the byte limit")
        metadata.update(body_complete=True, body_bytes=size)
        (directory / "http.json").write_text(json.dumps(metadata) + "\n")
        return 0
    except Exception as error:
        # Exception text can contain request data. Keep only its type here.
        (directory / "transport-error.json").write_text(
            json.dumps(dict(error_type=type(error).__name__)) + "\n"
        )
        return 1
    finally:
        if connection is not None:
            connection.close()


if __name__ == "__main__":
    raise SystemExit(exchange(Path(sys.argv[1]), float(sys.argv[2])))

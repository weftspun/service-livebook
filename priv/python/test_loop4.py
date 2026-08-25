"""Negative controls for loop 4's latent arm.

The arm's whole job is to refuse in the cases where refusing is the correct answer, so
every case here is a refusal that must happen. Two positive controls sit at the end, so a
module that refused everything could not pass.

No service is contacted. `requests` is replaced with a fake whose replies are written in
the test, which is the only way to exercise "the server said stub" without running a
server that says it.

    python priv/python/test_loop4.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import loop4  # noqa: E402
from weft_loop import PreconditionFailed  # noqa: E402

FAILS: list[str] = []


class FakeResponse:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise loop4.requests.RequestException(f"status {self.status_code}")

    def json(self):
        return self._payload


class FakeRequests:
    """Scripted replies by URL suffix, plus the exception type the real module exports."""

    RequestException = Exception

    def __init__(self, health=None, predict=None, extract=None, raises=None):
        self.health, self.predict, self.extract, self.raises = health, predict, extract, raises
        self.calls = []

    def get(self, url, timeout=None):
        self.calls.append(("get", url))
        if self.raises:
            raise self.raises
        return FakeResponse(self.health)

    def post(self, url, json=None, timeout=None):
        self.calls.append(("post", url))
        if url.endswith("/extract"):
            return FakeResponse(self.extract)
        return FakeResponse(self.predict)


def with_requests(fake, fn):
    real = loop4.requests
    loop4.requests = fake
    try:
        return fn()
    finally:
        loop4.requests = real


def expect_raise(label, fn, needle=None):
    try:
        fn()
    except PreconditionFailed as error:
        if needle and needle not in str(error):
            FAILS.append(label)
            print(f"  BAD {label}: refused for the wrong reason: {error}")
            return
        print(f"  ok  {label}: refused ({str(error)[:64]})")
        return
    except Exception as error:  # noqa: BLE001
        FAILS.append(label)
        print(f"  BAD {label}: raised {type(error).__name__}, wanted PreconditionFailed")
        return
    FAILS.append(label)
    print(f"  BAD {label}: accepted")


def expect_ok(label, fn):
    try:
        fn()
    except Exception as error:  # noqa: BLE001
        FAILS.append(label)
        print(f"  BAD {label}: raised {type(error).__name__}: {error}")
        return
    print(f"  ok  {label}: accepted")


def main() -> int:
    tmp = Path(tempfile.mkdtemp())
    source = tmp / "source.png"
    source.write_bytes(b"\x89PNG\r\n\x1a\n")
    b64_glb = loop4.base64.b64encode(b"glTF-stub").decode()
    b64_layer = loop4.base64.b64encode(b"#usda 1.0\n").decode()

    print("negative controls: each of these must refuse")

    expect_raise(
        "a stubbed VoxHammer is unavailable, not a successful repair",
        lambda: with_requests(
            FakeRequests(health={"status": "ok", "ready": True, "stub": True}),
            lambda: loop4.require_voxhammer("http://voxhammer"),
        ),
        needle="stub=true",
    )
    expect_raise(
        "a VoxHammer that is up but not loaded is unavailable",
        lambda: with_requests(
            FakeRequests(health={"status": "ok", "ready": False, "stub": False}),
            lambda: loop4.require_voxhammer("http://voxhammer"),
        ),
        needle="not ready",
    )
    expect_raise(
        "a VoxHammer that does not answer is unavailable",
        lambda: with_requests(
            FakeRequests(raises=Exception("connection refused")),
            lambda: loop4.require_voxhammer("http://voxhammer"),
        ),
        needle="/health",
    )
    expect_raise(
        "a working VoxHammer still stops, because a USD layer cannot be scored",
        lambda: with_requests(
            FakeRequests(
                health={"status": "ok", "ready": True, "stub": False},
                extract={"glb": b64_glb, "layer": b64_layer},
                predict={"layer": b64_layer, "plan": ["a_mark_region"], "seed": 42},
            ),
            lambda: loop4.repair_latent("http://voxhammer", "http://pixal3d", "STATE",
                                        str(source), tmp),
        ),
        needle="cannot be scored",
    )

    print("positive controls: a module that refused everything would pass the above")

    def health_passes():
        status = with_requests(
            FakeRequests(health={"status": "ok", "ready": True, "stub": False}),
            lambda: loop4.require_voxhammer("http://voxhammer"),
        )
        assert status["ready"] is True

    expect_ok("a ready, unstubbed VoxHammer passes the gate", health_passes)

    def extract_writes_both():
        glb, layer = with_requests(
            FakeRequests(extract={"glb": b64_glb, "layer": b64_layer}),
            lambda: loop4.extract("http://pixal3d", "STATE", tmp),
        )
        assert glb.read_bytes() == b"glTF-stub"
        assert layer.read_bytes().startswith(b"#usda")

    expect_ok("extract writes the glb and the USD layer", extract_writes_both)

    def region_is_the_whole_object():
        decoded = json.loads(loop4.base64.b64decode(loop4.region()))
        assert decoded == loop4.WHOLE_OBJECT

    expect_ok("the default region says the whole object rather than implying a choice",
              region_is_the_whole_object)

    def order_is_health_then_extract_then_edit():
        fake = FakeRequests(
            health={"status": "ok", "ready": True, "stub": False},
            extract={"glb": b64_glb, "layer": b64_layer},
            predict={"layer": b64_layer},
        )
        try:
            with_requests(fake, lambda: loop4.repair_latent(
                "http://voxhammer", "http://pixal3d", "STATE", str(source), tmp))
        except PreconditionFailed:
            pass
        assert [c[1] for c in fake.calls] == [
            "http://voxhammer/health",
            "http://pixal3d/extract",
            "http://voxhammer/predict",
        ], fake.calls

    expect_ok("the arm extracts a mesh before the editor sees it", order_is_health_then_extract_then_edit)

    print(f"\n{len(FAILS)} failed")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())

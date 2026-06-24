"""
End-to-end smoke test for the offline ECG analysis pipeline.

It does NOT check clinical correctness — that is done by reading the actual traces.
It only verifies that, on a clean machine, the detector installs, runs on a bundled
sample recording, and returns a physiologically plausible number of beats (i.e. the
pipeline is wired up and not silently broken).

Run directly:   python tests/test_smoke.py
Or via pytest:  pytest -q
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "samples" / "ecg_30s.csv"
DETECTOR = ROOT / "host" / "analyze_recording.py"

# ~30 s of ECG at a normal-ish heart rate (50-120 bpm) is roughly 25-60 beats.
# A wide window keeps the test robust to small threshold tweaks while still
# catching real failures (zero beats, a crash, or a nonsensical count).
MIN_BEATS, MAX_BEATS = 20, 70


def run_detector():
    """Run the detector on a temp copy of the sample so the output PNG never
    lands in the repo. Returns the detector's stdout."""
    assert SAMPLE.exists(), f"missing sample recording: {SAMPLE}"
    assert DETECTOR.exists(), f"missing detector script: {DETECTOR}"

    with tempfile.TemporaryDirectory() as tmp:
        local = Path(tmp) / SAMPLE.name
        shutil.copyfile(SAMPLE, local)
        env = dict(os.environ, MPLBACKEND="Agg")  # headless: no display needed
        proc = subprocess.run(
            [sys.executable, str(DETECTOR), str(local)],
            capture_output=True, text=True, env=env, timeout=120,
        )
    assert proc.returncode == 0, (
        f"detector exited with {proc.returncode}\n--- stderr ---\n{proc.stderr}"
    )
    return proc.stdout


def parse_total_beats(out):
    m = re.search(r"total beats:\s*(\d+)", out)
    assert m, f"could not find a beat count in detector output:\n{out}"
    return int(m.group(1))


def test_smoke():
    out = run_detector()
    assert "loaded" in out.lower(), "detector did not report loading the sample"
    n = parse_total_beats(out)
    assert MIN_BEATS <= n <= MAX_BEATS, (
        f"implausible beat count for ~30 s of ECG: {n} "
        f"(expected {MIN_BEATS}-{MAX_BEATS})"
    )


if __name__ == "__main__":
    test_smoke()
    print("OK: smoke test passed")

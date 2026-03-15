#!/usr/bin/env python3
"""Health monitoring: check site, APIs, predictions, disk.

Cron:
    */15 * * * * cd /var/www/orbit && .venv/bin/python3 scripts/health_check.py >> logs/health.log 2>&1
"""
import json
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("health")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PREDICTIONS_FILE = PROJECT_ROOT / "public" / "predictions.json"
LOGS_DIR = PROJECT_ROOT / "logs"


def check_site():
    """Verify orbitalchaos.online returns 200."""
    try:
        import urllib.request
        req = urllib.request.Request(
            "https://orbitalchaos.online",
            headers={"User-Agent": "HealthCheck/1.0"}
        )
        resp = urllib.request.urlopen(req, timeout=10)
        return (True, "Site OK") if resp.status == 200 else (False, f"HTTP {resp.status}")
    except Exception as e:
        return False, f"Site unreachable: {e}"


def check_iss_api():
    """Verify ISS tracking API responds."""
    try:
        import urllib.request
        resp = urllib.request.urlopen(
            "https://api.wheretheiss.at/v1/satellites/25544", timeout=10
        )
        data = json.loads(resp.read())
        return (True, "ISS API OK") if "latitude" in data else (False, "Bad format")
    except Exception as e:
        return False, f"ISS API: {e}"


def check_noaa_api():
    """Verify NOAA SWPC API responds."""
    try:
        import urllib.request
        resp = urllib.request.urlopen(
            "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json", timeout=10
        )
        data = json.loads(resp.read())
        return (True, "NOAA OK") if len(data) > 1 else (False, "No data")
    except Exception as e:
        return False, f"NOAA: {e}"


def check_predictions():
    """Verify predictions.json exists and is fresh."""
    if not PREDICTIONS_FILE.exists():
        return False, "predictions.json missing"
    try:
        with open(PREDICTIONS_FILE) as f:
            data = json.load(f)
        gen = data.get("generated_at", "")
        if not gen:
            return False, "No generated_at field"
        generated_at = datetime.fromisoformat(gen.replace("Z", "+00:00"))
        age = datetime.now(timezone.utc) - generated_at
        if age > timedelta(hours=2):
            return False, f"Stale ({age})"
        path = data.get("path", [])
        if len(path) < 5:
            return False, f"Too few points ({len(path)})"
        return True, f"OK (age: {age}, {len(path)} pts)"
    except Exception as e:
        return False, f"Invalid: {e}"


def check_disk():
    """Verify >1 GB free disk space."""
    try:
        stat = os.statvfs("/")
        free_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)
        return (True, f"{free_gb:.1f} GB free") if free_gb > 1.0 else (False, f"Low: {free_gb:.1f} GB")
    except Exception as e:
        return False, f"Disk check: {e}"


def check_pipeline():
    """Verify data pipeline ran in last 36 hours."""
    log_patterns = ["fetch*.log", "cron*.log"]
    newest_mtime = 0
    for pattern in log_patterns:
        for f in LOGS_DIR.glob(pattern):
            mtime = os.path.getmtime(f)
            if mtime > newest_mtime:
                newest_mtime = mtime
    if newest_mtime == 0:
        return False, "No fetch logs found"
    age = datetime.now() - datetime.fromtimestamp(newest_mtime)
    return (True, f"Last run {age} ago") if age < timedelta(hours=36) else (False, f"Stale ({age})")


def main():
    checks = [
        ("site", check_site),
        ("iss_api", check_iss_api),
        ("noaa_api", check_noaa_api),
        ("predictions", check_predictions),
        ("disk", check_disk),
        ("pipeline", check_pipeline),
    ]

    failures = []
    for name, fn in checks:
        ok, msg = fn()
        if ok:
            log.info(f"PASS {name}: {msg}")
        else:
            log.warning(f"FAIL {name}: {msg}")
            failures.append((name, msg))

    if failures:
        log.error(f"{len(failures)} check(s) failed")
        return 1

    log.info("All checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

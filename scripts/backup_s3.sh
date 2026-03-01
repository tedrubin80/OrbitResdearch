#!/bin/bash
# Full backup of Orbital Chaos project to S3 + local volume
# Usage: bash /var/www/orbit/scripts/backup_s3.sh
# Cron: weekly Sunday at 1 AM

set -euo pipefail

PROJECT_DIR="/var/www/orbit"
LOCAL_BACKUP_DIR="/mnt/HC_Volume_103339423/backups/orbit"
LOG_DIR="${PROJECT_DIR}/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/backup_${TIMESTAMP}.log"

mkdir -p "${LOG_DIR}" "${LOCAL_BACKUP_DIR}"

# Load S3 credentials
if [ -f "${PROJECT_DIR}/.env" ]; then
    export $(grep -v '^#' "${PROJECT_DIR}/.env" | xargs)
fi

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "${LOG_FILE}"
}

log "=== Orbital Chaos Backup Started ==="

# ── 1. Local tarball backup ──────────────────────────────────────────────────
log "Creating local backup tarball..."

TARBALL="${LOCAL_BACKUP_DIR}/orbit_backup_${TIMESTAMP}.tar.gz"

tar czf "${TARBALL}" \
    -C /var/www \
    --exclude='orbit/.venv' \
    --exclude='orbit/.git' \
    --exclude='orbit/__pycache__' \
    --exclude='orbit/logs' \
    orbit/

TARBALL_SIZE=$(du -sh "${TARBALL}" | cut -f1)
log "Local tarball: ${TARBALL} (${TARBALL_SIZE})"

# Keep only last 4 weekly backups locally
cd "${LOCAL_BACKUP_DIR}"
ls -t orbit_backup_*.tar.gz 2>/dev/null | tail -n +5 | xargs -r rm -f
KEPT=$(ls orbit_backup_*.tar.gz 2>/dev/null | wc -l)
log "Local backups retained: ${KEPT}"

# ── 2. Git backup ────────────────────────────────────────────────────────────
log "Git backup..."
cd "${PROJECT_DIR}"

if git diff --quiet && git diff --cached --quiet && [ -z "$(git ls-files --others --exclude-standard)" ]; then
    log "No git changes to commit"
else
    git add --ignore-errors src/ scripts/ notebooks/ paper/ public/ hf_space/ config.yaml requirements.txt .gitignore 2>/dev/null || true
    if ! git diff --cached --quiet; then
        git commit -m "Weekly backup - $(date +%Y-%m-%d)

Automated backup of code, configs, notebooks, and paper.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
        git push origin main 2>&1 | tee -a "${LOG_FILE}"
        log "Git pushed to origin/main"
    else
        log "No meaningful git changes"
    fi
fi

# ── 3. S3 backup ─────────────────────────────────────────────────────────────
log "Uploading to S3..."

# Activate venv for boto3
source "${PROJECT_DIR}/.venv/bin/activate"

python3 << 'PYEOF'
import os
import sys
import boto3
from botocore.config import Config
from pathlib import Path
from datetime import datetime

project = Path("/var/www/orbit")
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

s3 = boto3.client(
    "s3",
    endpoint_url=os.environ["S3_ENDPOINT"],
    aws_access_key_id=os.environ["S3_ACCESS_KEY"],
    aws_secret_access_key=os.environ["S3_SECRET_KEY"],
    region_name=os.environ.get("S3_REGION", "hel1"),
    config=Config(signature_version="s3v4", s3={"multipart_chunksize": 8 * 1024 * 1024}),
)

# Multipart transfer config for large files
transfer_config = boto3.s3.transfer.TransferConfig(
    multipart_threshold=8 * 1024 * 1024,
    max_concurrency=4,
    multipart_chunksize=8 * 1024 * 1024,
)

bucket = os.environ.get("S3_BUCKET", "orbit-research")

# Ensure bucket exists
try:
    s3.head_bucket(Bucket=bucket)
except Exception:
    s3.create_bucket(Bucket=bucket, CreateBucketConfiguration={"LocationConstraint": os.environ.get("S3_REGION", "hel1")})
    print(f"Created bucket: {bucket}")

uploaded = 0
errors = 0

# Upload code, configs, notebooks, paper, public, hf_space, scripts, checkpoints, data/raw
dirs_to_backup = [
    "src", "scripts", "notebooks", "paper", "public", "hf_space",
    "checkpoints", "data/raw",
]
files_to_backup = [
    "config.yaml", "requirements.txt", ".gitignore",
]

for d in dirs_to_backup:
    dir_path = project / d
    if not dir_path.exists():
        continue
    for f in dir_path.rglob("*"):
        if f.is_file() and "__pycache__" not in str(f):
            key = f"backup/{timestamp}/{f.relative_to(project)}"
            try:
                s3.upload_file(str(f), bucket, key, Config=transfer_config)
                uploaded += 1
            except Exception as e:
                print(f"  ERROR uploading {f}: {e}", file=sys.stderr)
                errors += 1

for fname in files_to_backup:
    fpath = project / fname
    if fpath.exists():
        key = f"backup/{timestamp}/{fname}"
        try:
            s3.upload_file(str(fpath), bucket, key, Config=transfer_config)
            uploaded += 1
        except Exception as e:
            print(f"  ERROR uploading {fpath}: {e}", file=sys.stderr)
            errors += 1

# Also upload the tarball
tarball = sorted(Path("/mnt/HC_Volume_103339423/backups/orbit").glob("orbit_backup_*.tar.gz"))
if tarball:
    latest = tarball[-1]
    key = f"backup/{timestamp}/{latest.name}"
    try:
        s3.upload_file(str(latest), bucket, key, Config=transfer_config)
        uploaded += 1
        print(f"Uploaded tarball: {latest.name}")
    except Exception as e:
        print(f"  ERROR uploading tarball: {e}", file=sys.stderr)
        errors += 1

# Cleanup old S3 backups (keep last 4)
response = s3.list_objects_v2(Bucket=bucket, Prefix="backup/", Delimiter="/")
prefixes = sorted([p["Prefix"] for p in response.get("CommonPrefixes", [])])
if len(prefixes) > 4:
    for old_prefix in prefixes[:-4]:
        # List and delete all objects under old prefix
        old_objects = s3.list_objects_v2(Bucket=bucket, Prefix=old_prefix)
        if "Contents" in old_objects:
            delete_keys = [{"Key": obj["Key"]} for obj in old_objects["Contents"]]
            s3.delete_objects(Bucket=bucket, Delete={"Objects": delete_keys})
            print(f"Cleaned old backup: {old_prefix}")

print(f"S3 upload complete: {uploaded} files, {errors} errors")
PYEOF

log "=== Backup Complete ==="

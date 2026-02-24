#!/bin/bash
# Setup cron job for automated orbit data fetching
# Runs daily at 2 AM, extending data to current date

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VENV="$PROJECT_DIR/.venv/bin/activate"
FETCH_SCRIPT="$PROJECT_DIR/scripts/cron_fetch.py"

# Cron entry: daily at 2 AM
CRON_ENTRY="0 2 * * * cd $PROJECT_DIR && source $VENV && python $FETCH_SCRIPT --extend >> $PROJECT_DIR/logs/cron.log 2>&1"

# Check if cron job already exists
if crontab -l 2>/dev/null | grep -q "cron_fetch.py"; then
    echo "Cron job already exists:"
    crontab -l | grep "cron_fetch.py"
    echo ""
    echo "To update, run: crontab -e"
else
    # Add to crontab
    (crontab -l 2>/dev/null; echo "$CRON_ENTRY") | crontab -
    echo "Cron job installed:"
    echo "$CRON_ENTRY"
fi

echo ""
echo "Verify with: crontab -l"
echo "Logs at: $PROJECT_DIR/logs/"

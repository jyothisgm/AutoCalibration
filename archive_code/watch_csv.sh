#!/usr/bin/env bash

CSV="theta_hat.csv"
TARGET_LINES=45

while true; do
    if [[ -f "$CSV" ]]; then
        lines=$(wc -l < "$CSV")
        echo "$(date) $CSV has $lines lines (waiting for $TARGET_LINES)..."

        if [[ "$lines" -eq "$TARGET_LINES" ]]; then
        echo "$(date) condition met. Running command..."
        python run_distributed.py
        break
        fi
    else
        echo "$(date) $CSV not found. Waiting..."
    fi

    sleep 300  # 5 minutes
done

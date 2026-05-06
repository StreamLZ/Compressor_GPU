#!/bin/bash
# StreamLZ benchmark script — downloads enwik8 if needed, runs -bc at t0, t1, t16.

DIR="$(cd "$(dirname "$0")" && pwd)"
SLZ="$DIR/streamlz"
ENWIK8="$DIR/enwik8"

if [ ! -f "$ENWIK8" ]; then
    echo "Downloading enwik8 (34 MB)..."
    curl -L -o "$DIR/enwik8.zip" https://mattmahoney.net/dc/enwik8.zip
    unzip -o "$DIR/enwik8.zip" -d "$DIR"
    rm -f "$DIR/enwik8.zip"
fi

echo
echo "=== Auto threads ==="
"$SLZ" -bc -r 30 -t 0 "$ENWIK8"

echo
echo "=== Single thread ==="
"$SLZ" -bc -r 30 -t 1 "$ENWIK8"

echo
echo "=== 16 threads ==="
"$SLZ" -bc -r 30 -t 16 "$ENWIK8"

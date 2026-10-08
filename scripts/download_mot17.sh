#!/usr/bin/env bash
# Download MOT17 (5.5 GB) or MOT20 (5.0 GB) into ./data
#   bash scripts/download_mot17.sh          # MOT17
#   bash scripts/download_mot17.sh MOT20    # MOT20
set -euo pipefail
NAME="${1:-MOT17}"
mkdir -p data
cd data
if [ ! -d "$NAME/train" ]; then
  echo "Downloading $NAME from motchallenge.net ..."
  curl -L --fail -o "$NAME.zip" "https://motchallenge.net/data/$NAME.zip" \
    || wget -O "$NAME.zip" "https://motchallenge.net/data/$NAME.zip"
  unzip -q "$NAME.zip"
  rm "$NAME.zip"
fi
echo "Sequences in data/$NAME/train:"
ls "$NAME/train"

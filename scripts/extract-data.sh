#!/bin/bash

# constants
CURRENT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$(dirname "$CURRENT_DIR")"
FILE_ID="13tWz6n6agPglhNl3uk74xVOefYuy471H"
RAW_DATA_PATH="$ROOT_DIR/data/raw"
FILENAME="data.zip"

# download the file
cd "$ROOT_DIR/data/raw"
gdown "$FILE_ID"

# extract the main zip file and remove zip itself
unzip -q "$RAW_DATA_PATH/$FILENAME" -d "$RAW_DATA_PATH"
rm "$RAW_DATA_PATH/$FILENAME"

echo "Extraction complete"
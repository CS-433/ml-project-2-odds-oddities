#!/bin/bash

# constants
CURRENT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$(dirname "$CURRENT_DIR")"
FILE_ID="1BPoDYytNB37pKZ1eWxSzVXXT9CgwrXV8"
MODEL_PATH="$ROOT_DIR/data/results/final_models"
FILENAME="models.zip"

# download the file
cd "$MODEL_PATH"
gdown "$FILE_ID"

# extract the main zip file and remove zip itself
unzip -q "$MODEL_PATH/$FILENAME" -d "$MODEL_PATH"
rm "$MODEL_PATH/$FILENAME"

echo "download complete"
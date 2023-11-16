#!/bin/bash

# constants
CURRENT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$(dirname "$CURRENT_DIR")"
ZIP_PATH="https://docs.google.com/uc?export=download&confirm=yes&id=1G0RizyPwZNSPMFBdtHjTw4MEqETdkzZw"
RAW_DATA_PATH="$ROOT_DIR/data/raw"
FILENAME="epfml-segmentation.zip"

# download the file
wget -O "$ROOT_DIR/data/raw/epfml-segmentation.zip" --no-check-certificate "$ZIP_PATH"

# extract the main zip file and remove zip itself
unzip -q "$RAW_DATA_PATH/$FILENAME" -d "$RAW_DATA_PATH"
rm "$RAW_DATA_PATH/$FILENAME"

# remove unnecessary files
find "$RAW_DATA_PATH" -type f -name "*.py" -exec rm {} \;

# extract the child zip
for subzip in "$RAW_DATA_PATH"/*.zip; do
    if [ -f "$subzip" ]; then
        unzip -q "$subzip" -d "$RAW_DATA_PATH"
        rm "$subzip"  # Optional: Remove the extracted sub-zip file
    fi
done

# flatten test
# TODO: rename 'test_set_images' to 'test' and 'training' to 'train'
for subdir in "$RAW_DATA_PATH/test_set_images"/*/; do
    if [ "$(ls -A "$subdir" | wc -l)" -eq 1 ]; then
        mv "${subdir}"* "$RAW_DATA_PATH/test_set_images/"
        rmdir "$subdir"
    fi
done

echo "Extraction complete"
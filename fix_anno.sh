#!/bin/bash

# Configuration
data_root='/nas/ai_image/sync_image/hazelnut/crack/'
train_ann_file='train/_annotations.coco.json'
train_data_prefix='train/'
val_ann_file='valid/_annotations.coco.json'
val_data_prefix='valid/'

# Function to rename images and update JSON annotations
rename_and_update() {
  ann_file="$1"
  data_prefix="$2"
  
  # Backup original annotation file
  cp "$data_root$ann_file" "$data_root$ann_file.bak"

  # Read image file names and rename them
  while IFS= read -r file; do
    new_file="$(uuidgen).${file##*.}"
    mv "$data_root$data_prefix$file" "$data_root$data_prefix$new_file"

    # Update JSON annotations
    sed -i "s/\"$file\"/\"$new_file\"/g" "$data_root$ann_file"
  done < <(jq -r ".images[].file_name" "$data_root$ann_file")
}

# Run the rename_and_update function for train and val sets
rename_and_update "$train_ann_file" "$train_data_prefix"
rename_and_update "$val_ann_file" "$val_data_prefix"


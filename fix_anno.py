import os
import json
import uuid

data_root = '/opt/images/hz/'
train_ann_file = 'train/_annotations.coco.json'
train_data_prefix = 'train/'
val_ann_file = 'valid/_annotations.coco.json'
val_data_prefix = 'valid/'

def rename_and_update(ann_file, data_prefix):
    with open(os.path.join(data_root, ann_file), 'r') as f:
        annotations = json.load(f)

    file_mapping = {}
    for image in annotations['images']:
        old_file = image['file_name']
        new_file = f"{uuid.uuid4()}.{old_file.split('.')[-1]}"
        os.rename(os.path.join(data_root, data_prefix, old_file), os.path.join(data_root, data_prefix, new_file))
        file_mapping[old_file] = new_file
        image['file_name'] = new_file

    with open(os.path.join(data_root, ann_file), 'w') as f:
        json.dump(annotations, f)

rename_and_update(train_ann_file, train_data_prefix)
rename_and_update(val_ann_file, val_data_prefix)


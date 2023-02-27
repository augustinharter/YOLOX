#%%
import json

#%% 
with open('/home/augo/data/COCO/annotations/instances_val2017.json', 'r') as f:
    coco = json.load(f)

# %%
import fiftyone as fo
import fiftyone.zoo as foz
# The directory containing the source images
data_path = "/home/augo/data/COCO/val2017"

# The path to the COCO labels JSON file
labels_path = "/home/augo/data/COCO/annotations/instances_val2017.json"

# Import the dataset
dataset = fo.Dataset.from_dir(
    dataset_type=fo.types.COCODetectionDataset,
    data_path=data_path,
    labels_path=labels_path,
)

session = fo.launch_app(dataset)
# %%
session.wait()

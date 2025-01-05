import os
from detectron2.data import DatasetCatalog, MetadataCatalog

def get_deepglobe_dicts(img_dir, mask_dir):
    from PIL import Image
    dataset_dicts = []
    for idx, file_name in enumerate(os.listdir(img_dir)):
        record = {}

        file_path = os.path.join(img_dir, file_name)
        mask_path = os.path.join(mask_dir, file_name.replace(".jpg", ".png"))

        # 假设图像和掩码尺寸一致
        record["file_name"] = file_path
        record["sem_seg_file_name"] = mask_path
        record["image_id"] = idx
        dataset_dicts.append(record)
    return dataset_dicts

def register_deepglobe_segmentation():
    for d in ["train", "val"]:
        img_dir = f"deepglobe/images/{d}"
        mask_dir = f"deepglobe/annotations/{d}"
        DatasetCatalog.register(
            f"deepglobe_{d}",
            lambda d=d: get_deepglobe_dicts(img_dir, mask_dir)
        )
        MetadataCatalog.get(f"deepglobe_{d}").set(
            stuff_classes=["background", "road"],  # 修改为你的类别名称
            ignore_label=255,
            evaluator_type="sem_seg"
        )

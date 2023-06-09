from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

class FasterRCNNDataset(torch.utils.data.Dataset):

    BBOX_AREA_THRESHOLD = 4

    def __init__(self,
                 scenes: list[Path],
                 transform: dict = None,
                 obj_ids: list = [1, 2, 3],
                 infer: bool = True
                 ):

        self.scene_paths = scenes
        self.infer = infer
        self.format = format
        self.show = True
        
        self.obj_ids = obj_ids
        self.obj_id_mappings = {str(obj_id): idx for idx, obj_id in enumerate(obj_ids)}
        
        self.scene_dict = self.__load_scenes_into_dict()
        
        self.transform = transform
    
    def extract_obj_id(self, obj_id, id_convention):
        
        if id_convention == 'object_poses':
            return self.obj_id_mappings[str(obj_id["obj_id"])]
        elif id_convention == 'object_ids':
            return self.obj_id_mappings[str(obj_id)]
        else:
            raise ValueError("id_convention must be either 'object_poses' or 'object_ids'.")
    
    def __getitem__(self, index):

        # Find all relevant file_paths (images, masks, yaml files)
        image_path = self.scene_dict[index]["images_path"]
        # mask_path  = self.scene_dict[index]["masks_path"]
        yaml_file  = self.scene_dict[index]["yaml_files"]

        # Load corresponding images, masks and camera configs
        image  = cv2.imread(str(image_path))
        image  = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # mask   = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        # Load scene data from yaml files
        d = self.__extract_yaml_info(yaml_file)

        # Construct all possible datapoints (x, y, w, h, img_id, instance_id, obj_id)
        bboxes = []
        id_convention = 'object_poses' if 'object_poses' in d else 'object_ids'
        for _, bb, obj_id in zip(d['mask_ids'], d['bounding_boxes'], d[id_convention]):
            if bb[2] * bb[3] > self.BBOX_AREA_THRESHOLD:
                bboxes.append([bb[0], bb[1], bb[2], bb[3], self.extract_obj_id(obj_id, id_convention)])

        # Apply augmentations
        labels = [0] * len(bboxes)
        transformed = self.transform(image=image, bboxes=bboxes, labels=labels)
        image, bboxes = transformed['image'], transformed['bboxes']

        if self.show:
            for bbox in bboxes:
                x, y, w, h, label = bbox
                x, y, w, h = int(x), int(y), int(w), int(h)
                if label == 0:
                    color = (255, 0, 0)
                elif label == 1:
                    color = (0, 255, 0)
                else:
                    color = (0, 0, 255)
                cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
            plt.imshow(image)
            plt.show()

        # Convert bboxes (x, y, w, h) to (x1, y1, x2, y2) and normalized to [0, 1]
        boxes = []
        labels = []
        for bbox in bboxes:
            x, y, w, h, obj_id = bbox
            x1, y1, x2, y2 = x, y, x + w, y + h
            x1, y1, x2, y2 = x1 / image.shape[1], y1 / image.shape[0], x2 / image.shape[1], y2 / image.shape[0]
            boxes.append([x1, y1, x2, y2])

        target = {}
        target["boxes"] = torch.Tensor(boxes).float()
        target["labels"] = torch.Tensor(labels).long()

        return image, target

    def __len__(self):        
        return len(self.scene_dict)
    
    def __load_scenes_into_dict(self):
        
        scene_dict = dict()
        global_idx = 0

        for local_idx in range(len(self.scene_paths)):
            
            image_paths, mask_paths, yaml_files = self.__file_paths(local_idx)

            indices = np.arange(len(image_paths))

            for local_idx in indices:

                scene_dict[global_idx] = {
                    "images_path": image_paths[local_idx],
                    "masks_path": mask_paths[local_idx],
                    "yaml_files": yaml_files[local_idx],
                }

                global_idx += 1

        return scene_dict

    def __file_paths(self, index):
            
        color_match, masks_match, img_match = "rgb/*", "mask_crushed/*", "img_*.yaml"

        images_path = sorted(list(self.scene_paths[index].glob(color_match)))
        masks_path = sorted(list(self.scene_paths[index].glob(masks_match)))
        yaml_files = sorted([x for x in self.scene_paths[index].glob(img_match)])
        
        if len(images_path) == 0:
            images_path = sorted(list(self.scene_paths[index].glob("color_*.png")))
            masks_path = sorted(list(self.scene_paths[index].glob("segm_*.png")))
            
        return images_path, masks_path, yaml_files

    def __extract_yaml_info(self, yaml_file):
            
        with open(yaml_file, 'r') as f:
            scene_data = yaml.load(f, Loader=yaml.FullLoader)
        
        return scene_data

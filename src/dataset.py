
import os, glob
from dataclasses import dataclass
from functools import lru_cache
import logging


@dataclass
class ImageInfo:
    path : str
    name : str
    target: int
    desc : str = "unknown"

class ImagenetSource:

    def __init__(self, selection_name=None):
        self.base_path = "/home/weziv5/work/data/imagenet"
        self.targets_path = os.path.join(self.base_path, "imagenet_validation_ground_truth.txt")
        self.selection_name = selection_name

    @lru_cache(maxsize=None)     
    def read_selection(self):
        selection_path = os.path.join(self.base_path, f"{selection}.txt")
        with open(selection_path, "rt") as sf:
            return [self.get_image_name(x.strip()) for x in sf]

    @lru_cache(maxsize=None)     
    def get_all_images(self):
        all_images = glob.glob(os.path.join(self.base_path, "validation", "/*.JPEG"))
        image_targets = self.get_image_targets()

        images = {}
        for path in all_images:
            image_name = self.get_image_name(path)
            images[image_name] = ImageInfo(
                path=path, 
                name=image_name,
                target=image_targets[image_name])
        

    @lru_cache(maxsize=None)
    def get_image_targets(self):
        rv = {}
        with open(self.targets_path, "rt") as tf:
            for line in tf:
                file_name, target = line.split()
                target = int(target)
                image_name = self.get_image_name(file_name)
                rv[image_name] = target
        return rv

    def get_image_name(self, path):
        image_file_name = os.path.basename(path)
        image_name = image_file_name[0:image_file_name.find(".")]
        return image_name    

        

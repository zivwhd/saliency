
import os, glob, random
from dataclasses import dataclass
from functools import lru_cache
import logging


@dataclass
class ImageInfo:
    path : str
    name : str
    target: int
    desc : str = "unknown"

BASE_IMAGENET_PATH="/home/weziv5/work/data/imagenet"

class ImagenetSource:

    def __init__(self, 
                 base_path=BASE_IMAGENET_PATH, 
                 image_dir_ptrn="validation",
                 selection_name=None):
        self.base_path = base_path
        self.image_dir_ptrn = image_dir_ptrn
        self.targets_path = os.path.join(self.base_path, "imagenet_validation_ground_truth.txt")
        self.selection_name = selection_name

    @lru_cache(maxsize=None)     
    def read_selection(self):
        logging.debug("loading selection")
        selection_path = os.path.join(self.base_path, f"{self.selection_name}.smp")
        with open(selection_path, "rt") as sf:
            return [self.get_image_name(x.strip()) for x in sf]

    @lru_cache(maxsize=None)     
    def get_all_images(self):
        logging.debug(f"imagenet base path: {self.base_path}")
        ptrn = os.path.join(self.base_path, self.image_dir_ptrn, "*.JPEG")
        all_images = glob.glob(ptrn)
        logging.debug(f"found {len(all_images)} images at {ptrn}")
        image_targets = self.get_image_targets()

        images = {}
        for path in all_images:
            image_name = self.get_image_name(path)
            images[image_name] = ImageInfo(
                path=path, 
                name=image_name,
                target=image_targets[image_name])

        if self.selection_name:
            selection = self.read_selection()
            images = {name : img for name, img in images.items() if name in selection}

        return images

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

        
class Coord:

    def __init__(self, items, base_path):
        self.items = items
        self.base_path = base_path
        self.iter_items = None
        self.iter_last_wip = None

    def __iter__(self):
        self.iter_last_wip = None
        self.iter_items = [] + self.items
        random.shuffle(self.iter_items)
        return self

    def mark_done(self):
        if self.iter_last_wip:
            wip_path, done_path = self.iter_last_wip
            os.rename(wip_path, done_path)            
            self.iter_last_wip = None

    def __next__(self):

        self.mark_done()

        while self.iter_items:
            item = self.iter_items.pop()
            name = item.name
            
            os.makedirs(self.base_path, exist_ok=True)

            rnd = random.randint(0,int(1e9))
            tmp_path = os.path.join(self.base_path, f"{name}.{rnd:x}.tmp")
            wip_path = os.path.join(self.base_path, f"{name}.wip")
            done_path = os.path.join(self.base_path, f"{name}.done")

            if os.path.isfile(done_path):
                continue

            with open(tmp_path) as tmp_file:
                pass

            try:
                os.rename(tmp_path, wip_path)
                self.iter_last_name = (wip_path, done_path)
                return item
            
            except OSError as e:        
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            
        self.mark_done()
        raise StopIteration


import os, glob, random
import errno
from dataclasses import dataclass
from functools import lru_cache
import logging
from PIL import Image
import torchvision
import atexit

@dataclass
class ImageInfo:
    path : str
    name : str
    target: int
    desc : str = "unknown"

BASE_IMAGENET_PATH="/home/weziv5/work/data/imagenet"

class ISource:
    @lru_cache(maxsize=None)     
    def read_selection(self):
        logging.debug("loading selection")
        selection_path = os.path.join(self.base_path, f"{self.selection_name}.smp")
        with open(selection_path, "rt") as sf:
            return [self.get_image_name(x.strip()) for x in sf]

    def get_image_file_ptrn(self):
        return "*.JPEG"
    
    @lru_cache(maxsize=None)     
    def get_all_images(self):
        logging.debug(f"imagenet base path: {self.base_path}")
        ptrn = os.path.join(self.base_path, self.image_dir_ptrn, self.get_image_file_ptrn())
        all_images = glob.glob(ptrn)
        logging.debug(f"found {len(all_images)} images at {ptrn}")
        image_targets = self.get_image_targets()

        images = {}
        for path in all_images:
            image_name = self.get_image_name(path)
            images[image_name] = ImageInfo(
                path=path, 
                name=image_name,
                target=image_targets.get(image_name,0))

        if self.selection_name:
            selection = self.read_selection()
            images = {name : img for name, img in images.items() if name in selection}

        return images

    def get_image_name(self, path):
        image_file_name = os.path.basename(path)
        image_name = image_file_name[0:image_file_name.find(".")]
        return image_name    


class VOCSource(ISource):
    def __init__(self,
                 base_path="/home/weziv5/work/data/voc/VOC2012_test", 
                 image_dir_ptrn="JPEGImages",
                 selection_name=None):
        self.base_path = base_path
        self.image_dir_ptrn = image_dir_ptrn
        self.selection_name = selection_name

    def get_image_targets(self):
        return {}

    def get_image_file_ptrn(self):
        return "*.jpg"

class ImagenetSource(ISource):

    def __init__(self, 
                 base_path=BASE_IMAGENET_PATH, 
                 image_dir_ptrn="validation",
                 selection_name=None):
        self.base_path = base_path
        self.image_dir_ptrn = image_dir_ptrn
        self.targets_path = os.path.join(self.base_path, "imagenet_validation_ground_truth.txt")
        self.selection_name = selection_name



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


class CustomImageNetDataset:

    def __init__(self, images):
        self.images = images
        self.targets = [x.target for x in images]
        #self.prune()

    def prune(self):
        bad_images = []
        logging.info("pruning bad images")
        for idx in range(len(self)):
            try:
                self[idx]
            except:
                logging.info(f"found bad image {idx} {self.images[idx].path}")
                bad_images.append(idx)

        self.images = [self.images[idx] for idx in range(len(self.images)) if idx not in bad_images]
        self.targets = [x.target for x in self.images]

        logging.info(f"Done pruning {len(bad_images)}")


    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        info = self.images[idx]
        #logging.debug(f"loading {info.path}")
        img = Image.open(info.path)
        shape = (224,224)
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(shape),
            torchvision.transforms.CenterCrop(shape),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])])

        tns = transform(img)
        return tns, info.target

class Coord:

    def __init__(self, items, base_path, getname=lambda x: x.name):
        self.items = items
        self.getname = getname
        self.base_path = base_path
        self.iter_items = None
        self.iter_last_wip = None
        atexit.register(self.cleanup)

    def __iter__(self):
        self.iter_last_wip = None
        self.iter_items = [] + self.items
        random.shuffle(self.iter_items)
        return self

    def cleanup(self):        
        if self.iter_last_wip:
            wip_path, done_path = self.iter_last_wip
            logging.info(f"cleanup {wip_path}")
            os.unlink(wip_path)
        

    def mark_done(self):
        try:
            if self.iter_last_wip:
                wip_path, done_path = self.iter_last_wip
                logging.debug(f"marking done {done_path}")
                os.rename(wip_path, done_path)            
                self.iter_last_wip = None
        except:
            logging.exception("failed marking done")

    def __next__(self):

        self.mark_done()

        while self.iter_items:
            item = self.iter_items.pop()
            name = self.getname(item)
            
            os.makedirs(self.base_path, exist_ok=True)
            
            rnd = random.randint(0,int(1e9))
            #tmp_path = os.path.join(self.base_path, f"{name}.{rnd:x}.tmp")
            wip_path = os.path.join(self.base_path, f"{name}.wip")
            done_path = os.path.join(self.base_path, f"{name}.done")

            if os.path.isfile(done_path):
                continue            

            os.makedirs(os.path.dirname(wip_path), exist_ok=True)

            if self.acquire(wip_path):
                self.iter_last_wip = (wip_path, done_path)
                logging.debug(f"acquired {wip_path}")
                logging.debug(f"handling {name}")
                return item
            else:
                logging.debug(f"skipping {wip_path}")

        self.mark_done()

        raise StopIteration

    def acquire(self, path):
        try:
            # Open file with O_CREAT (create) and O_EXCL (fail if exists)
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR)
            #print(f"File {path} created successfully.")
            os.close(fd)
            return True
        except OSError as e:
            if e.errno == errno.EEXIST:
                #print(f"File {filename} already exists.")
                return False
            else:
                raise  # For other types of OSError



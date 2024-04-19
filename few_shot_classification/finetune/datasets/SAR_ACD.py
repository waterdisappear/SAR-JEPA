import os
import pickle
from scipy.io import loadmat
import re
from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import mkdir_if_missing

from .oxford_pets import OxfordPets


@DATASET_REGISTRY.register()
class SAR_ACD(DatasetBase):

    dataset_dir = "SAR_ACD"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.split_path = os.path.join(self.dataset_dir, "split_Li_SAR_ACD.json")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)

        if os.path.exists(self.split_path):
            train, val, test = OxfordPets.read_split(self.split_path, self.dataset_dir)
        else:
            trainval_file = os.path.join(self.dataset_dir, 'images')
            # test_file = os.path.join(self.dataset_dir, 'images')
            trainval = self.read_data(trainval_file)
            # test = self.read_data(test_file)
            train, test_val = OxfordPets.split_trainval(trainval, p_val=0.9)
            # OxfordPets.save_split(train, val, test, self.split_path, self.dataset_dir)

        num_shots = cfg.DATASET.NUM_SHOTS
        if num_shots >= 1:
            seed = cfg.SEED
            preprocessed = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl")
            
            if os.path.exists(preprocessed):
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    train, val = data["train"], data["val"]
            else:
                train = self.generate_fewshot_dataset(train, num_shots=num_shots)
                val = self.generate_fewshot_dataset(test_val, num_shots=min(num_shots*1, 10))
                data = {"train": train, "val": val}
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

        subsample = cfg.DATASET.SUBSAMPLE_CLASSES
        train, val, test = OxfordPets.subsample_classes(train, val, test_val, subsample=subsample)

        super().__init__(train_x=train, val=val, test=test)

    def read_data(self, image_dir):
        label_int = {'A220': 0, 'A330': 1, 'ARJ21': 2, 'Boeing737': 3, 'Boeing787': 4}

        # label_name = {'BMP2': 'BMP2',
        #               'BTR70': 'BTR70',
        #               'T72': 'T72',
        #               'BTR60': 'BTR60',
        #               '2S1': '2S1',
        #               'BRDM2': 'BRDM2',
        #               'D7': 'D7',
        #               'T62': 'T62',
        #               'ZIL131': 'ZIL131',
        #               'ZSU234': 'ZSU234'}

        label_name = {'A220': 'A220', 'A330': 'A330', 'ARJ21': 'ARJ21', 'Boeing737': 'Boeing737', 'Boeing787': 'Boeing787'}

        items = []

        for root, dirs, files in os.walk(image_dir):
            files = sorted(files)
            for file in files:
                if os.path.splitext(file)[1] == '.jpg':
                    impath = os.path.join(root, file)
                    idx = re.split('[/\\\]', impath).index('SAR_ACD')
                    label = label_int[re.split('[/\\\]', impath)[idx+2]]
                    classname = label_name[re.split('[/\\\]', impath)[idx+2]]
                    item = Datum(impath=impath, label=label, classname=classname)
                    items.append(item)
        return items

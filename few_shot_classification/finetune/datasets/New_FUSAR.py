import os
import pickle
from scipy.io import loadmat
import re
from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import mkdir_if_missing

from .oxford_pets import OxfordPets


@DATASET_REGISTRY.register()
class New_FUSAR(DatasetBase):

    dataset_dir = "New_FUSAR"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.split_path = os.path.join(self.dataset_dir, "split_Li_SAR_ACD.json")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)

        if os.path.exists(self.split_path):
            train, val, test = OxfordPets.read_split(self.split_path, self.dataset_dir)
        else:
            trainval_file = os.path.join(self.dataset_dir, 'Train')
            test_file = os.path.join(self.dataset_dir, 'Val')
            trainval = self.read_data(trainval_file)
            test = self.read_data(test_file)
            train, val = OxfordPets.split_trainval(trainval)
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
                val = self.generate_fewshot_dataset(val, num_shots=min(num_shots*1, 10))
                data = {"train": train, "val": val}
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

        subsample = cfg.DATASET.SUBSAMPLE_CLASSES
        train, val, test = OxfordPets.subsample_classes(train, val, test, subsample=subsample)

        super().__init__(train_x=train, val=val, test=test)

    def read_data(self, image_dir):
        label_int = {'Bridges': 0, 'Cargo': 1, 'CoastalLands_island': 2, 'Fishing': 3, 'LandPatches': 4, 'OtherShip': 5,
                     'SeaClutterWaves': 6, 'SeaPatches': 7, 'StrongFalseAlarms': 8, 'Tanker': 9}

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

        label_name = {'Bridges': 'Bridges', 'Cargo': 'Cargo', 'CoastalLands_island': 'CoastalLands_island',
                      'Fishing': 'Fishing', 'LandPatches': 'LandPatches', 'OtherShip': 'OtherShip',
                      'SeaClutterWaves': 'SeaClutterWaves', 'SeaPatches': 'SeaPatches',
                      'StrongFalseAlarms': 'StrongFalseAlarms', 'Tanker': 'Tanker'}

        # label_name = {'BMP2': 'BMP2, a type of Infantry Fighting Vehicle',
        #               'BTR70': 'BTR70, a type of Armored Personnel Carrier',
        #               'T72': 'T72, a heavy tank is sitting in a field',
        #               'BTR60': 'BTR60, a military personnel carrier is seen from above in a field of grass and dirt',
        #               '2S1': '2S1, a self-propelled artillery is laying in the middle of a field',
        #               'BRDM2': 'BRDM2, a military scout car is in the middle of a field of mud and dirt',
        #               'D7': 'D7, a bulldozer is sitting in a field with a bulldozer blade',
        #               'T62': 'T62, a tank is sitting in a field',
        #               'ZIL131': 'ZIL131, a military truck parked in a field',
        #               'ZSU234': 'ZSU234, a Self-propelled Anti-aircraft Gun with a satellite dish on top of it in a field of dirt and grass'}

        # turet, amour,barrels, auxiliary machine guns, hatches, tracks, periscopes, amphibious, the number of wheels/track, and whether they have two side doors
        # label_name = {
        #     'BMP2': 'BMP2, a type of armored amphibious infantry fighting vehicle (a two-man turret, an auxiliary machine gun, a missile launcher, track vehicle, 14.3 tonnes, 6.85m length, 3.08 width, 2.07m height)',
        #     'BRDM2': 'BRDM2, a type of amphibious armored patrol car (a heavy machine gun and a general-purpose machine gun, boat-like bow, 4 wheels, side doors, 7.7 tonnes, 5.75m length, 2.37m width, 2.31 height)',
        #     'BTR60': 'BTR60, a type of eight-wheeled armored personnel carrier (a hevay machine gun and a machine gun, amphibious, two side doors, 10.3 tonnes, 7.56 length, 2.83m width, 2.31m height )',
        #     'BTR70': 'BTR70, a type of eight-wheeled armored personnel carrier (a hevay machine gun and a machine gun, two-piece escape, amphibious, side troop doors, 11.5 tonnes, 7.53 length, 2.80m width, 2.32m height )',
        #     'T62': 'T62, a type of medium main battle Tank (a 115mm smoothbore gun, armored, two machine guns, 6 periscope viewports, track vehicle, 10 wheels, 40 tonnes, 9.34m length, 3.30m width, 2.40 height)',
        #     'T72': 'T72, a type of heavy main battle tank (a 125mm smoothbore gun, armored, two machine guns, small periscope viewports, track vehicle, 41.5 tonnes, 9.53m length, 3.59m width, 2.23 height)',
        #     '2S1': '2S1, a type of self-propelled howitzer (a howitzer, armored, amphibious, track vehicle, 14 wheels, 16 tonnes, 7.26m length, 2.85m width, 2.73 height)',
        #     'D7': 'D7, a type of medium bulldozer (a blade， 1.4 tonnes, 4.1m length, 2.5m width, 2.4 height)',
        #     'ZIL131': 'ZIL131, a type of army truck (4 wheels, 3.5 tons, 7.04m length, 2.49m width, 2.4 height)',
        #     'ZSU234': 'ZSU234, a type of self-propelled anti-aircraft weapon system (low rectangular turret with side bulges, lightly armored, 4 cannon, 6 wheels, 20.5 tons, 6.54m length, 2.95m width, 2.25 height)'}

        items = []

        for root, dirs, files in os.walk(image_dir):
            files = sorted(files)
            for file in files:
                if os.path.splitext(file)[1] == '.jpg' or os.path.splitext(file)[1] == '.png':
                    impath = os.path.join(root, file)
                    idx = re.split('[/\\\]', impath).index('New_FUSAR')
                    label = label_int[re.split('[/\\\]', impath)[idx+2]]
                    classname = label_name[re.split('[/\\\]', impath)[idx+2]]
                    item = Datum(impath=impath, label=label, classname=classname)
                    items.append(item)
        return items

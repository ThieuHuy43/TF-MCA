import torchvision
from .dataset_base import DatasetBase
import os

class MiniImagenet(DatasetBase):

    def __init__(self, root):
        super(MiniImagenet, self).__init__(root=root, name='miniimagenet')

        self.root = root
        self.classes = CLASSES

        self.image_folder = os.path.join(self.root, 'miniimagenet/images')
        self.split_folder = os.path.join(self.root, 'miniimagenet/split') 
        # self.csv_path = os.path.join(self.csv_folder, f'{split_name}.csv')


        self.train_data, self.train_targets = self.load_data_targets_from_csv(os.path.join(self.split_folder, 'train.csv'))
        self.test_data, self.test_targets = self.load_data_targets_from_csv(os.path.join(self.split_folder, 'test.csv'))

        self.gpt_prompt_path = r"description/modified_miniimagent_output_final.json"

    def load_data_targets_from_csv(self, csv_path):
        data = []
        targets = []
        class2index = dict()
        data2target = dict()
        with open(csv_path, 'r') as f:
            lines = f.readlines()[1:]
            for line in lines:
                path, label = line.strip().split(',')
                full_path = os.path.join(self.image_folder, path)
                data.append(full_path)
                if label not in class2index:
                    class2index[label] = len(class2index)
                targets.append(class2index[label])
                data2target[full_path] = class2index[label]
        return data, targets



    def get_class_name(self):
        return self.classes
    
    def get_train_data(self):
        return self.train_data, self.train_targets
    
    def get_test_data(self):
        return self.test_data, self.test_targets



CLASSES = ['house finch', 'robin', 'triceratops', 'green mamba', 'harvestman', 
           'toucan', 'goose', 'jellyfish', 'nematode', 'king crab', 'dugong', 
           'Walker hound', 'Ibizan hound', 'Saluki', 'golden retriever', 'Gordon setter', 
           'komondor', 'boxer', 'Tibetan mastiff', 'French bulldog', 'malamute', 'dalmatian', 
           'Newfoundland', 'miniature poodle', 'white wolf', 'African hunting dog', 'Arctic fox', 
           'lion', 'meerkat', 'ladybug', 'rhinoceros beetle', 'ant', 'black-footed ferret', 
           'three-toed sloth', 'rock beauty', 'aircraft carrier', 'ashcan', 'barrel', 'beer bottle', 
           'bookshop', 'cannon', 'carousel', 'carton', 'catamaran', 'chime', 'clog', 'cocktail shaker', 
           'combination lock', 'crate', 'cuirass', 'dishrag', 'dome', 'electric guitar', 'file', 'fire screen', 
           'frying pan', 'garbage truck', 'hair slide', 'holster', 'horizontal bar', 'hourglass', 'iPod', 'lipstick', 
           'miniskirt', 'missile', 'mixing bowl', 'oboe', 'organ', 'parallel bars', 'pencil box', 'photocopier',
            'poncho', 'prayer rug', 'reel', 'school bus', 'scoreboard', 'slot', 'snorkel', 'solar dish', 
            'spider web', 'stage', 'tank', 'theater curtain', 'tile roof', 'tobacco shop', 'unicycle', 'upright', 
            'vase', 'wok', 'worm fence', 'yawl', 'street sign', 'consomme', 'trifle', 'hotdog', 'orange', 'cliff', 
            'coral reef', 'bolete', 'ear']

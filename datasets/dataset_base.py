from abc import  abstractmethod


class DatasetBase:

    def __init__(self, root, name):
        self.root = root
        self.name = name
        self.template = ['a photo of a {}.']

        # self.template = ["itap of a {}.",
        #                 "a bad photo of the {}.",
        #                 "a origami {}.",
        #                 "a photo of the large {}.",
        #                 "a {} in a video game.",
        #                 "art of the {}.",
        #                 "a photo of the small {}."]

    
    @abstractmethod
    def get_class_name(self):
        raise NotImplementedError()

    @abstractmethod 
    def get_train_data(self):
        raise NotImplementedError()

    @abstractmethod
    def get_test_data(self):
        raise NotImplementedError()
    
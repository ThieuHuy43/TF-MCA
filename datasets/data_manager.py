import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class DatasetManager:

    def __init__(self, cfg):

        # Properties
        self.cfg = cfg

        # Dataset split setting
        self.root = cfg.DATASET.ROOT
        self.dataset_name    = cfg.DATASET.NAME
        self.num_init_cls    = cfg.DATASET.NUM_INIT_CLS
        self.num_inc_cls     = cfg.DATASET.NUM_INC_CLS
        self.num_base_shot   = cfg.DATASET.NUM_BASE_SHOT
        self.num_inc_shot    = cfg.DATASET.NUM_INC_SHOT
        
        # training setting of data
        self.num_workers     = cfg.DATALOADER.NUM_WORKERS
        self.train_batchsize_base = cfg.DATALOADER.TRAIN.BATCH_SIZE_BASE
        self.train_batchsize_inc = cfg.DATALOADER.TRAIN.BATCH_SIZE_INC
        self.test_batchsize = cfg.DATALOADER.TEST.BATCH_SIZE

        # setup data
        self._setup_data(self.root, self.dataset_name)
        self.class_index_in_task = []
        self.class_index_in_task.append(np.arange(0, self.num_init_cls))
        for start in range(self.num_init_cls, self.num_total_classes, self.num_inc_cls):
            end = min(start + self.num_inc_cls, self.num_total_classes)
            self.class_index_in_task.append(np.arange(start, end))
        self.num_tasks = len(self.class_index_in_task)
        self.train_transform, self.test_transform = self._set_transform()
        print(self.class_index_in_task)



    def _setup_data(self, root, dataset_name):
        full_dataset = get_data_source(root, dataset_name)
        self.class_names = full_dataset.classes
        self.template = full_dataset.template
        self.train_data, self.train_targets = full_dataset.get_train_data()
        self.test_data, self.test_targets = full_dataset.get_test_data()

        # convert labels  to `np.ndarray` for convenient indexing
        if not isinstance(self.train_targets, np.ndarray):
            self.train_targets = np.array(self.train_targets)
        if not isinstance(self.test_targets, np.ndarray):
            self.test_targets = np.array(self.test_targets)
        
        self.num_total_classes = len(self.class_names)
    

    def get_dataset(self, task_id, source, mode=None, accumulated_past=False):
        '''
        source: which part of dataset
        mode: which data transform is used
        accumulated_past (Bool): Whether the training data in this contains the data from the past 
        '''
        assert 0 <= task_id < len(self.class_index_in_task), \
               f"task id {task_id} should be in range [0, {len(self.class_index_in_task) - 1}]"

        # Get data
        if source == 'train':
            # When training, using data of task [i]
            x, y = self.train_data, self.train_targets
            if accumulated_past:
                class_idx = np.concatenate(self.class_index_in_task[0: task_id + 1])
            else:
                class_idx = self.class_index_in_task[task_id]

        elif source == 'test':
            # When testing, using data of tasks [0..i]
            x, y = self.test_data, self.test_targets
            class_idx = np.concatenate(self.class_index_in_task[0: task_id + 1])

        else:
            raise ValueError(f'Invalid data source :{source}')
        
        # Get Transform
        if mode == 'train':
            transform = self.train_transform
        elif mode == 'test':
            transform = self.test_transform
        else:
            raise ValueError(f'Invalid transform mode: {mode}')

        def find_sublist_indices(matrix, numbers):
            """
            Function to find the indices of the sublists where each number in 'numbers' is located.

            Parameters:
            matrix (list of list of int): The 2D list to search in.
            numbers (np.ndarray): The numpy array of numbers to search for.

            Returns:
            dict: A dictionary with keys as the numbers from 'numbers' and values as the indices of the sublists.
            """
            indices = {}
            for x in numbers:
                found = False
                for i, sublist in enumerate(matrix):
                    if x in sublist:
                        indices[x] = i
                        found = True
                        break
                if not found:
                    indices[x] = -1  # If number not found, set index to -1
            return indices
        
        class_to_task_id = find_sublist_indices(self.class_index_in_task, class_idx)
        num_shot = self.num_base_shot if task_id == 0 else self.num_inc_shot
        data, targets = self._select_data_from_class_index(x, y, class_idx, num_shot, source)
        task_dataset = TaskDataset(data, targets, transform, class_to_task_id, self.class_names)
        return task_dataset
    

    
    def get_dataloader(self, task_id, source, mode=None, accumulate_past=False):
        assert source in ['train', 'test'], f'data source must be in ["train", "test"], got {source}'
        # the default mode is same as source
        if mode == None:
            mode = source
        dataset = self.get_dataset(task_id, source, mode, accumulate_past)
        if source == 'train':
            if task_id == 0:
                batchsize = self.train_batchsize_base
            else:
                batchsize = self.train_batchsize_inc
            loader = DataLoader(dataset,
                                batch_size=batchsize,
                                shuffle=False,
                                num_workers=self.num_workers,
                                drop_last=False,
                                pin_memory=True)
        elif source == 'test':
            loader = DataLoader(dataset,
                                batch_size=self.test_batchsize,
                                shuffle=False,
                                num_workers=self.num_workers,
                                drop_last=False,
                                pin_memory=True)
        else:
            raise ValueError(f'Invalid data source: {source}')
        return loader
    


    def _select_data_from_class_index(self, x, y, class_idx, shot, source):
        ret_x = []
        ret_y = []
        if isinstance(x, list):
            x = np.array(x)
        for c in class_idx:
            idx_c = np.where(y == c)[0]
            
            if shot is not None and source == 'train':
                # Random choosing index
                # NOTE: Only when training, we can modify the num of samples
                # assert shot <= len(idx_c), f"shot {shot} should not be greater than {len(idx_c)}"
                if shot == -1:
                    idx_selected = idx_c
                
                elif shot > len(idx_c):
                    # num of shot is greater than num of samples in this class
                    # hence use all samples in this class
                    print(f'shot:{shot} is greater than num of sample:{len(idx_c)} in class{c}')
                    idx_selected = idx_c
                else:
                    idx_selected = np.random.choice(idx_c, size=shot, replace=False)
            else:
                idx_selected = idx_c

            #ret_x.append(x[idx_selected])
            #ret_y.append(y[idx_selected])
            ret_x.append(np.array(x)[idx_selected])
            ret_y.append(np.array(y)[idx_selected])
        ret_x = np.concatenate(ret_x)
        ret_y = np.concatenate(ret_y)

        return ret_x, ret_y
    

    def _set_transform(self):
        img_size = 224
        MEAN = [0.48145466, 0.4578275, 0.40821073]
        STD  = [0.26862954, 0.26130258, 0.27577711]
        train_transform  = transforms.Compose([
            # transforms.RandomResizedCrop(img_size, scale=(0.5, 1), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomResizedCrop((img_size, img_size), scale=(0.08, 1.0), ratio=(0.75, 1.333), interpolation=transforms.InterpolationMode.BICUBIC, antialias=None),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
        test_transform = transforms.Compose([
            transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
        return train_transform, test_transform
    

class TaskDataset(Dataset):
    def __init__(self, images, labels, transform, class_to_task_id=None, class_name=None):
        self.images = images
        self.labels = labels
        self.transform = transform
        self.class_to_task_id = class_to_task_id
        self.class_name = class_name
        
        # Detect if images are paths or tensors
        if len(images) > 0:
            sample = images[0]
            self.use_path = isinstance(sample, str)  # Check if it's file path
            print(f"Dataset mode: {'Path-based' if self.use_path else 'Tensor-based'}")
        else:
            self.use_path = False

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.use_path:
            # Load image from path on-demand
            try:
                image = self.transform(pil_loader(self.images[idx]))
            except Exception as e:
                print(f"Error loading {self.images[idx]}: {e}")
                # Fallback: create dummy black image
                dummy_img = Image.new('RGB', (224, 224), (0, 0, 0))
                image = self.transform(dummy_img)
        else:
            # Handle pre-loaded tensors (legacy)
            img = self.images[idx]
            
            if img.ndim == 3 and img.shape[0] == 3:
                img = np.transpose(img, (1, 2, 0))

            img = img.astype(np.uint8)
            img = Image.fromarray(img).convert("RGB")
            image = self.transform(img)

        label = self.labels[idx]
        
        # Determine task_id and class_name
        if self.class_to_task_id is not None:
            task_id = self.class_to_task_id[label]
        else:
            task_id = -1
        
        if self.class_name is not None:
            cls_name = self.class_name[label]
        else:
            cls_name = ''
        
        # Return dict format like original code expects
        ret = {
            'idx': idx, 
            'image': image,
            'label': label,
            'cls_name': cls_name,
            'task_id': task_id
        }
        return ret

def pil_loader(path):
    """
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    """
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


# NEED MODIFY HERE IF YOU WANT TO ADD NEW DATASETS
def get_data_source(root, name):
    from .cifar100 import CIFAR100
    from .miniimagenet import MiniImagenet
    from .cub200 import CUB200
    from .fruits import Fruits
    source_dict = {
        'cifar100' : CIFAR100,
        'miniimagenet' : MiniImagenet,
        'cub200': CUB200,
        'fruits': Fruits
    }
    return source_dict[name.lower()](root=root)
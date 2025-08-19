import os
import glob
import numpy as np
from PIL import Image
from .dataset_base import DatasetBase

class Fruits(DatasetBase):
    def __init__(self, root, session_id=0, train_split=0.8):
        super(Fruits, self).__init__(root=root, name='fruits')
        
        self.data_path = "tf_fruit"
        
        # ✅ Simple template for now
        self.template = ['a photo of a {}.']
        
        # ✅ Load paths thay vì images - Memory efficient
        self.train_data, self.train_targets, self.test_data, self.test_targets = self.load_data_paths()
        
        # Load class names
        self.classes = self.get_class_names_from_folders()
        
        self.gpt_prompt_path = 'description/fruits_prompts_full.json'
        
        print(f"Dataset loaded: {len(self.train_data)} training, {len(self.test_data)} testing")
        print(f"Number of classes: {len(self.classes)}")

    def load_data_paths(self):
        """
        ✅ Load file paths thay vì images - Memory efficient như CUB200
        """
        train_data = []      # ✅ Chứa đường dẫn file
        train_targets = []   # ✅ Chứa class index
        test_data = []       # ✅ Chứa đường dẫn file  
        test_targets = []    # ✅ Chứa class index
        
        # Get all class folders
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data path {self.data_path} not found!")
            
        class_folders = sorted([d for d in os.listdir(self.data_path) 
                               if os.path.isdir(os.path.join(self.data_path, d))])
        
        print(f"Found {len(class_folders)} classes")
        
        for class_idx, class_name in enumerate(class_folders):
            class_path = os.path.join(self.data_path, class_name)
            
            # ✅ Get all image paths in class
            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', 
                              '*.JPG', '*.JPEG', '*.PNG', '*.BMP', '*.TIFF']
            image_paths = []
            for ext in image_extensions:
                image_paths.extend(glob.glob(os.path.join(class_path, ext)))
            
            if len(image_paths) == 0:
                print(f"Warning: No images found in {class_path}")
                continue
                
            print(f"Class {class_name}: {len(image_paths)} images")
            
            # ✅ NO quality selection for now - use all images
            # Split all available images
            np.random.seed(42)  # Reproducible
            indices = np.random.permutation(len(image_paths))
            train_split_idx = int(0.8 * len(image_paths))
            
            train_indices = indices[:train_split_idx]
            test_indices = indices[train_split_idx:]
            
            # ✅ Add ALL paths (no quality filtering)
            for i in train_indices:
                train_data.append(image_paths[i])      # ✅ Path only
                train_targets.append(class_idx)
                
            for i in test_indices:
                test_data.append(image_paths[i])       # ✅ Path only
                test_targets.append(class_idx)
        
        print(f"Total: {len(train_data)} train paths, {len(test_data)} test paths")
        return train_data, train_targets, test_data, test_targets

    def get_class_names_from_folders(self):
        """✅ Get class names from folder structure"""
        class_folders = sorted([d for d in os.listdir(self.data_path) 
                               if os.path.isdir(os.path.join(self.data_path, d))])
        return class_folders

    def get_class_name(self):
        """✅ Return class names"""
        return self.classes

    def get_train_data(self):
        """✅ Return paths và targets - Memory efficient"""
        return self.train_data, self.train_targets

    def get_test_data(self):
        """✅ Return paths và targets - Memory efficient"""
        return self.test_data, self.test_targets
# import os
# from torchvision import datasets, transforms
# from torch.utils.data import Dataset, random_split
# from .dataset_base import DatasetBase

# class Fruits(DatasetBase):
#     def __init__(self, root, session_id=0, train_split=0.8):
#         super(Fruits, self).__init__(root=root, name='fruits')
        
#         self.data_path = "tf_fruit"
        
#         # Simplified transforms without augmentation
#         self.transform = transforms.Compose([
#             transforms.Resize((200, 200)),  # CLIP requires 224x224
#             # transforms.Resize((192, 192)),  # CLIP requires 224x224
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                               std=[0.229, 0.224, 0.225])  # ImageNet stats
#         ])

#         # Load dataset
#         full_dataset = datasets.ImageFolder(
#             self.data_path,
#             transform=self.transform
#         )
        
#         # Calculate splits
#         total_size = len(full_dataset)
#         train_size = int(train_split * total_size)
#         test_size = total_size - train_size
        
#         # Split dataset
#         self.train_dataset, self.test_dataset = random_split(
#             full_dataset, 
#             [train_size, test_size]
#         )

#         # Store class information
#         self.classes = full_dataset.classes
#         self.class_to_idx = full_dataset.class_to_idx
        
#         self.gpt_prompt_path = 'description/fruits_prompts_full.json'
        
#         print(f"Dataset loaded: {len(self.train_dataset)} training, {len(self.test_dataset)} testing")
#         print(f"Number of classes: {len(self.classes)}")

#     def get_class_name(self):
#         return self.classes

#     def get_train_data(self):
#         imgs = []
#         labels = []
#         for img, label in self.train_dataset:
#             imgs.append(img)
#             labels.append(label)
#         return imgs, labels

#     def get_test_data(self):
#         imgs = []
#         labels = []
#         for img, label in self.test_dataset:
#             imgs.append(img)
#             labels.append(label)
#         return imgs, labels

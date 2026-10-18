from PIL import Image
import torch.utils.data as data
import os

class SteelDataset(data.Dataset):
    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.img_sets = os.listdir(self.path)

    def __getitem__(self, idx):
        img_name = self.img_sets[idx]
        img_path = os.path.join(self.path, img_name)
        label = self.label_dir
        image = Image.open(img_path)
        return label, image

    def __len__(self):
        return len(self.img_sets)

root_dir = r'D:\Documents\DEV\Python\datasets\steel\casting_data\train'
ok_label_dir = 'ok_front'
ng_label_dir = 'def_front'
ok_dataset = SteelDataset(root_dir, ok_label_dir)
ng_dataset = SteelDataset(root_dir, ng_label_dir)

print(len(ok_dataset))
l, i = ok_dataset[0]
print(l)
i.show()

# merge dataset
trn_dataset = ok_dataset + ng_dataset
print(len(trn_dataset))


import os
from skimage import io
from torch.utils.data import Dataset


class CelebADataSet(Dataset):
    def __init__(self, root=None, transform=None, attr='All', is_attr=True):
        self.root_dir = root
        self.data_path = os.path.join(self.root_dir, 'img_align_celeba')
        self.attr_path = os.path.join(self.root_dir, 'list_attr_celeba.txt')
        self.transform = transform
        self.attr = attr
        self.is_attr = is_attr
        self.transform = transform
        self.image_names = self.get_attr()

    def get_attr(self):
        image_names = []
        if self.attr == 'All':
            self.data_set_size = len(os.listdir(self.data_path))
            return image_names
        with open(self.attr_path, 'r') as f:
            f.readline()
            attribute_names = f.readline().strip().split(' ')
            self.attribute_names_list = list(attribute_names)
            attr_list = [attribute_names.index(attr) for attr in [self.attr]]
            self.attr_index = attr_list[0]
            for i, line in enumerate(f):
                fields = line.strip().replace('  ', ' ').split(' ')
                img_name = fields[0]
                if int(img_name[:6]) != i + 1:
                    raise ValueError('Parse error.')
                if fields[self.attr_index + 1] == str(-1 + 2 * int(self.is_attr)):
                    image_names.append(fields[0])
            self.data_set_size = len(image_names)
        return image_names

    def __len__(self):
        return self.data_set_size

    def __getitem__(self, idx):
        if self.attr == 'All':
            img_name = os.path.join(self.data_path, '%.6d.jpg' % (idx + 1))
        else:
            img_name = os.path.join(self.data_path, self.image_names[idx])
        image = io.imread(img_name)
        return self.transform(image)

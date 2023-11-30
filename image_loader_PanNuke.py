import os
from PIL import Image
import joint_transforms
import torch.utils.data as data
from torchvision import transforms

# test_img_transform = transforms.Compose([
#     # transforms.Resize((512, 512)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                          std=[0.229, 0.224, 0.225])
# ])
# transform_mask = transforms.Compose([
#     transforms.Resize((512, 512)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.406],
#                          std=[0.225])
# ])

joint_transform = joint_transforms.Compose([
    joint_transforms.RandomHorizontallyFlip(),
    joint_transforms.RandomRotate(90),
    joint_transforms.RandomCrop(256),  # TNBC/PanNuke
    # joint_transforms.Resize((256, 256))  # GlaS
])

val_joint_transform = joint_transforms.Compose([
    joint_transforms.Resize((256, 256)),  # GlaS/PanNuke
    # joint_transforms.Resize((512, 512))  # TNBC
])

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

label_transform = transforms.ToTensor()
to_pil = transforms.ToPILImage()


# TODO 此处对于不同的数据集，修改不同的后缀
def make_dataset(root):
    # lis = os.listdir(os.path.join(root, 'image'))
    img_list = [os.path.splitext(f)[0] for f in os.listdir(os.path.join(root, 'images')) if f.endswith('.jpg') or f.endswith('.png')]
    return [
        (os.path.join(root, 'images', img_name + '.png'), os.path.join(root, 'masks', img_name + '.png'))
        for img_name in img_list]


class ImageFolder(data.Dataset):
    def __init__(self, root, mode='train', joint_transform=None, img_transform=None, label_transform=None):
        self.root = root
        self.imgs = make_dataset(root)
        self.joint_transform = joint_transform
        self.img_transform = img_transform
        self.label_transform = label_transform
        self.mode = mode

    def __getitem__(self, index):
        img_path, gt_path = self.imgs[index]
        img = Image.open(img_path).convert('RGB')
        target = Image.open(gt_path)

        if self.mode == 'train':
            img, target = self.joint_transform(img, target)  # for training, both im and label are resized
        else:
            img, target = self.joint_transform(img, target)  # for testing, both im and label are resized, but no other trans
        img = self.img_transform(img)  # to tensor and normalized
        target = self.label_transform(target)

        return img, target, img_path

    def __len__(self):
        return len(self.imgs)

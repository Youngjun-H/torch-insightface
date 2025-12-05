import numbers
import os
import warnings

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder

# class MXFaceDataset(Dataset):
#     """MXNet RecordIO 형식 데이터셋"""

#     def __init__(self, root_dir: str):
#         super().__init__()
#         self.transform = transforms.Compose(
#             [
#                 transforms.ToPILImage(),
#                 transforms.RandomHorizontalFlip(),
#                 transforms.ToTensor(),
#                 transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
#             ]
#         )
#         self.root_dir = root_dir
#         path_imgrec = os.path.join(root_dir, "train.rec")
#         path_imgidx = os.path.join(root_dir, "train.idx")

#         self.imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, "r")
#         s = self.imgrec.read_idx(0)
#         header, _ = mx.recordio.unpack(s)

#         if header.flag > 0:
#             self.header0 = (int(header.label[0]), int(header.label[1]))
#             self.imgidx = np.array(range(1, int(header.label[0])))
#         else:
#             self.imgidx = np.array(list(self.imgrec.keys))

#     def __getitem__(self, index):
#         idx = self.imgidx[index]
#         s = self.imgrec.read_idx(idx)
#         header, img = mx.recordio.unpack(s)
#         label = header.label

#         if not isinstance(label, numbers.Number):
#             label = label[0]
#         label = torch.tensor(label, dtype=torch.long)

#         sample = mx.image.imdecode(img).asnumpy()
#         if self.transform is not None:
#             sample = self.transform(sample)

#         return sample, label

#     def __len__(self):
#         return len(self.imgidx)


class SyntheticDataset(Dataset):
    """테스트용 합성 데이터셋"""

    def __init__(self):
        super().__init__()
        img = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.int32)
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).squeeze(0).float()
        img = ((img / 255) - 0.5) / 0.5
        self.img = img
        self.label = 1

    def __getitem__(self, index):
        return self.img, self.label

    def __len__(self):
        return 1000000


def get_dataset(root_dir: str) -> Dataset:
    """
    데이터셋 타입을 자동으로 감지하여 적절한 Dataset 반환

    Args:
        root_dir: 데이터셋 루트 디렉토리

    Returns:
        Dataset 인스턴스
    """
    # Synthetic 데이터셋
    if root_dir == "synthetic":
        return SyntheticDataset()

    # ImageFolder 형식
    # ArcFace 표준: 112x112 이미지 크기
    transform = transforms.Compose(
        [
            transforms.Resize((112, 112)),  # 모든 이미지를 112x112로 리사이즈
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    return ImageFolder(root_dir, transform=transform)

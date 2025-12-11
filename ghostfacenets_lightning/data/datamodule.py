"""
Lightning DataModule for Face Recognition
"""

import lightning as L
from data.dataset import FaceDataset, get_train_transform
from torch.utils.data import DataLoader


class FaceDataModule(L.LightningDataModule):
    """Lightning DataModule for face recognition"""

    def __init__(
        self,
        data_dir: str,
        batch_size: int = 128,
        num_workers: int = 4,
        input_size: int = 112,
        random_status: int = 2,
        val_split: float = 0.1,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.input_size = input_size
        self.random_status = random_status
        self.val_split = val_split

        self.train_dataset = None
        self.val_dataset = None
        self.num_classes = None

    def prepare_data(self):
        """Prepare data - calculate num_classes before setup
        This is called automatically by Lightning before setup()
        """
        import os

        if os.path.exists(self.data_dir):
            classes = sorted(
                [
                    d
                    for d in os.listdir(self.data_dir)
                    if os.path.isdir(os.path.join(self.data_dir, d))
                ]
            )
            self.num_classes = len(classes)
            print(f"Found {self.num_classes} classes in dataset")
        else:
            raise ValueError(f"Data directory does not exist: {self.data_dir}")

    def setup(self, stage=None):
        """Setup datasets - automatically called by Lightning"""
        if stage == "fit" or stage is None:
            # Training dataset only
            # Validation is handled by verification callbacks (LFW, AgeDB-30, etc.)
            self.train_dataset = FaceDataset(
                self.data_dir,
                transform=get_train_transform(self.input_size, self.random_status),
                is_train=True,
            )

            # Verify num_classes matches
            if self.num_classes is None:
                self.num_classes = self.train_dataset.num_classes
            elif self.num_classes != self.train_dataset.num_classes:
                print(
                    f"Warning: num_classes mismatch. prepare_data: {self.num_classes}, "
                    f"dataset: {self.train_dataset.num_classes}"
                )
                self.num_classes = self.train_dataset.num_classes

    def train_dataloader(self):
        """Train dataloader"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        """Validation dataloader - disabled, using verification callbacks instead"""
        # Return None to disable Lightning's validation loop
        # Verification is handled by FaceVerificationCallback
        return None

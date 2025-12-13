"""
Lightning DataModule for Face Recognition
"""

import lightning as L
from data.dataset import FaceDataset, get_train_transform
from torch.utils.data import ConcatDataset, DataLoader


class FaceDataModule(L.LightningDataModule):
    """Lightning DataModule for face recognition"""

    def __init__(
        self,
        data_dir: str | list[str],
        batch_size: int = 128,
        num_workers: int = 4,
        input_size: int = 112,
        random_status: int = 2,
        val_split: float = 0.1,
    ):
        super().__init__()
        # data_dir를 리스트로 변환 (단일 경로도 지원)
        if isinstance(data_dir, str):
            self.data_dirs = [data_dir]
        else:
            self.data_dirs = data_dir
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

        all_classes = set()
        for data_dir in self.data_dirs:
            if not os.path.exists(data_dir):
                raise ValueError(f"Data directory does not exist: {data_dir}")
            classes = sorted(
                [
                    d
                    for d in os.listdir(data_dir)
                    if os.path.isdir(os.path.join(data_dir, d))
                ]
            )
            all_classes.update(classes)
            print(f"Found {len(classes)} classes in {data_dir}")

        self.num_classes = len(all_classes)
        print(f"Total unique classes across all datasets: {self.num_classes}")

    def setup(self, stage=None):
        """Setup datasets - automatically called by Lightning"""
        if stage == "fit" or stage is None:
            # 여러 데이터셋을 합치기
            datasets = []
            all_class_to_idx = {}
            current_idx = 0

            for data_dir in self.data_dirs:
                # 각 데이터셋의 클래스를 스캔하여 통합 클래스 매핑 생성
                import os
                classes = sorted(
                    [
                        d
                        for d in os.listdir(data_dir)
                        if os.path.isdir(os.path.join(data_dir, d))
                    ]
                )

                # 클래스 오프셋 계산 (이미 존재하는 클래스는 기존 ID 사용)
                class_offsets = {}
                for class_name in classes:
                    if class_name not in all_class_to_idx:
                        all_class_to_idx[class_name] = current_idx
                        current_idx += 1
                    class_offsets[class_name] = all_class_to_idx[class_name]

                # 각 데이터셋에 대해 최소 오프셋 계산
                if class_offsets:
                    min_offset = min(class_offsets.values())
                    # 하지만 각 클래스마다 다른 오프셋이 필요하므로 직접 재매핑 필요
                    # 대신 FaceDataset을 수정하여 클래스별 오프셋을 지원하거나
                    # 여기서 샘플을 직접 재매핑

                # 임시로 각 데이터셋을 로드하고 클래스 ID 재매핑
                temp_dataset = FaceDataset(
                    data_dir,
                    transform=None,  # transform은 나중에 적용
                    is_train=True,
                )

                # 클래스 ID 재매핑
                remapped_samples = []
                for img_path, label in temp_dataset.samples:
                    original_class = temp_dataset.idx_to_class[label]
                    new_label = all_class_to_idx[original_class]
                    remapped_samples.append((img_path, new_label))

                # 재매핑된 샘플을 가진 데이터셋 생성
                class RemappedDataset(FaceDataset):
                    def __init__(self, samples, transform, num_classes):
                        self.samples = samples
                        self.transform = transform
                        self.num_classes = num_classes
                        self.is_train = True
                        self.class_offset = 0

                    def __getitem__(self, idx):
                        img_path, label = self.samples[idx]
                        # Load image
                        try:
                            from PIL import Image
                            image = Image.open(img_path).convert("RGB")
                        except Exception as e:
                            print(f"Error loading {img_path}: {e}")
                            image = Image.new("RGB", (112, 112), (0, 0, 0))
                        # Apply transforms
                        if self.transform:
                            image = self.transform(image)
                        return image, label

                remapped_dataset = RemappedDataset(
                    remapped_samples,
                    get_train_transform(self.input_size, self.random_status),
                    len(all_class_to_idx),
                )
                datasets.append(remapped_dataset)
                print(f"Added dataset from {data_dir}: {len(remapped_samples)} samples")

            # 여러 데이터셋 합치기
            if len(datasets) == 1:
                self.train_dataset = datasets[0]
            else:
                self.train_dataset = ConcatDataset(datasets)
                print(f"Combined {len(datasets)} datasets: {len(self.train_dataset)} total samples")

            # Verify num_classes matches
            if self.num_classes is None:
                self.num_classes = len(all_class_to_idx)
            elif self.num_classes != len(all_class_to_idx):
                print(
                    f"Warning: num_classes mismatch. prepare_data: {self.num_classes}, "
                    f"dataset: {len(all_class_to_idx)}"
                )
                self.num_classes = len(all_class_to_idx)

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

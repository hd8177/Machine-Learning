from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from typing import Tuple, List

def create_transforms(image_size: int = 64):
    """Match your notebook: augmentation for train, plain resize for test/val."""
    train_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ToTensor()
    ])

    test_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])

    return train_transforms, test_transforms


def create_dataloaders_from_single_folder(
    data_dir: str,
    image_size: int = 64,
    batch_size: int = 32,
    train_split: float = 0.8,
    num_workers: int = 2,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, List[str]]:
    """
    Matches your notebook logic:
    - Use ONE folder (data/train)
    - Create two ImageFolder datasets with different transforms
    - Split each into train + val using the SAME split sizes
    """
    train_tfms, test_tfms = create_transforms(image_size=image_size)

    # Two datasets pointing at same folder, different transforms (exactly like your cell 13)
    train_data_base = datasets.ImageFolder(root=data_dir, transform=train_tfms)
    test_data_base  = datasets.ImageFolder(root=data_dir, transform=test_tfms)

    class_names = train_data_base.classes

    # Split sizes
    total_len = len(train_data_base)
    train_len = int(total_len * train_split)
    val_len = total_len - train_len

    # Deterministic split
    generator = __import__("torch").Generator().manual_seed(seed)

    train_dataset, _ = random_split(train_data_base, [train_len, val_len], generator=generator)
    _, val_dataset   = random_split(test_data_base,  [train_len, val_len], generator=generator)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, class_names


def create_test_transform(image_size: int = 64):
    """Matches your later inference cells (cell 33): resize + ToTensor."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])

"""BaseDataset — abstract base class for all datasets.

Standardized data loading pipeline:
    download(save_dir) -> path
    preprocess() -> train/val/test splits
    __len__() / __getitem__(idx)
    get_dataloader(batch_size, ...) -> DataLoader
    split(ratios) -> (train, val, test)

Metadata properties:
    dataset_name / dataset_url / feature_info / num_users / num_items
"""
# TODO: Implement BaseDataset(ABC, Dataset) with abstract methods

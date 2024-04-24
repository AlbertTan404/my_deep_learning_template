from torch.utils.data import Dataset
from src.utils import setup_logger


logger = setup_logger(__name__)


class MyDataset(Dataset):
    def __init__(self, dataset_dir: str, split: str, epoch_scaling: int=1) -> None:
        super().__init__()
        raise NotImplementedError

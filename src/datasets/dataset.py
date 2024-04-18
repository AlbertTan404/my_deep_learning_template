from torch.utils.data import Dataset
from src.utils import setup_logger


logger = setup_logger(__name__)


class MyDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        raise NotImplementedError

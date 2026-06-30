"""
HeightmapDataset - единый формат загрузки данных для всего проекта.
"""

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

TERRAIN_TO_IDX = {"flat": 0, "hilly": 1, "mountain": 2}
IDX_TO_TERRAIN = {v: k for k, v in TERRAIN_TO_IDX.items()}
NUM_TERRAIN_CLASSES = len(TERRAIN_TO_IDX)


class HeightmapDataset(Dataset):
    """Загружает один split (train/val/test) heightmap-плиток.

    Параметры:
    processed_dir: путь к data/processed
    metadata_csv: путь к data/metadata.csv
    split: 'train' | 'val' | 'test'
    return_label: если True, __getitem__ возвращает (heightmap, label), иначе только heightmap.
    """

    def __init__(
        self,
        processed_dir: str = "data/processed",
        metadata_csv: str = "data/metadata.csv",
        split: str = "train",
        return_label: bool = False,
    ):
        super().__init__()
        assert split in ("train", "val", "test"), f"Неизвестный split: {split}"
        self.split = split
        self.return_label = return_label

        processed_dir = Path(processed_dir)
        tensor_path = processed_dir / f"{split}.pt"
        if not tensor_path.exists():
            raise FileNotFoundError(
                f"Не найден {tensor_path}. Сначала запусти prepare_dataset.py "
                f"или make_dummy_data.py."
            )
        self.data = torch.load(tensor_path, map_location="cpu")
        if self.data.dtype != torch.float32:
            self.data = self.data.float()
        meta = pd.read_csv(metadata_csv)
        self.meta = meta[meta["split"] == split].reset_index(drop=True)
        if len(self.meta) != len(self.data):
            raise ValueError(
                f"Рассогласование для split={split}: "
                f"{len(self.data)} патчей в .pt, но {len(self.meta)} строк "
                f"в metadata.csv. Контракт нарушен."
            )
        self.labels = torch.tensor(
            [TERRAIN_TO_IDX[t] for t in self.meta["terrain_type"]],
            dtype=torch.long,
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        x = self.data[idx]  # [1, 256, 256]
        if self.return_label:
            return x, self.labels[idx]
        return x

    def denormalize(self, x_norm: torch.Tensor, idx: int) -> torch.Tensor:
        """Перевод нормализованного патча [-1,1] обратно в метры по строке idx.
        """
        row = self.meta.iloc[idx]
        lo, hi = float(row["orig_min"]), float(row["orig_max"])
        return (x_norm + 1.0) / 2.0 * (hi - lo) + lo


if __name__ == "__main__":
    for split in ("train", "val", "test"):
        try:
            ds = HeightmapDataset(split=split, return_label=True)
        except FileNotFoundError as e:
            print(e)
            break
        x, y = ds[0]
        counts = {
            IDX_TO_TERRAIN[i]: int((ds.labels == i).sum()) for i in range(NUM_TERRAIN_CLASSES)
        }
        print(
            f"[{split}] N={len(ds)} | sample={tuple(x.shape)} "
            f"dtype={x.dtype} range=[{x.min():.2f},{x.max():.2f}] | классы={counts}"
        )

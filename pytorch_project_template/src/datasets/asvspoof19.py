from pathlib import Path
import numpy as np
import torch
import librosa
from tqdm.auto import tqdm
from typing import Dict, List, Any, Literal
from hydra.utils import instantiate

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import read_json, write_json

class ASVSpoof19(BaseDataset):
    """
    Dataset class for ASVSpoof 2019 LA partition.
    """

    def __init__(self, name: str ="train", data_dir=None, *args, **kwargs):
        if data_dir is None:
            self.data_dir: Path = Path("/kaggle/input/datasets/awsaf49/asvpoof-2019-dataset/LA/LA")
            # for using in google colab:
            # self.data_dir: Path = Path("/content/asvpoof2019/LA/LA")
        else:
            self.data_dir: Path = Path(data_dir)

        self.name: str = name
    
        index_path: Path = Path(f"/kaggle/working/cache/asvspoof19_{name}_index.json")
        # for using in google colab:
        # index_path: Path = Path(f"/cache/asvspoof19_{name}_index.json")

        if index_path.exists():
            index: Any = read_json(str(index_path))
        else:
            index: List[Dict] = self._create_index(name, index_path)

        super().__init__(index, *args, **kwargs)

    def _create_index(self, name: str, index_path: Path) -> List[Dict]:
        index: List[Dict] = []

        protocol_fnames: Dict[str, str] = {"train": "ASVspoof2019.LA.cm.train.trn.txt",
            "dev": "ASVspoof2019.LA.cm.dev.trl.txt",
            "eval": "ASVspoof2019.LA.cm.eval.trl.txt",
        }

        if name not in protocol_fnames:
            raise ValueError("Variable name must be 'train', 'dev' or 'eval'.")

        audio_dir: Path = self.data_dir / f"ASVspoof2019_LA_{name}" / "flac"
        protocol_path: Path = self.data_dir / "ASVspoof2019_LA_cm_protocols" / protocol_fnames[name]

        with open(protocol_path) as f:
            lines: List[str] = f.readlines()

        for line in tqdm(lines, desc=f"[ASVSpoof19] Parsing {name} protocol"):
            parts: List[str] = line.strip().split()

            if len(parts) <= 4:
                continue

            speaker: str = parts[0]
            audio_file: str = parts[1]
            label_str: str = parts[4]

            label: Literal[1, 0] = 1 if label_str == "bonafide" else 0

            audio_path: Path = audio_dir / f"{audio_file}.flac"

            index.append(
                {
                    "path": str(audio_path),
                    "label": label,
                    "speaker_id": speaker,
                    "audio_name": audio_file,
                }
            )

        index_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(index, str(index_path))

        return index

    def load_object(self, path: Path) -> Any:
        """
        Loading object with torchaudio.
        
        Args:
            path (Path): path of file (.flac).
        Returns:
            data_object (Tensor): audio signal (1-dimensional).
        """
        audio, sr = librosa.load(path, sr=None, mono=True)
        audio = torch.from_numpy(audio).float()
        return audio

    def __getitem__(self, idx: int):
        item = self._index[idx]
        data_object = self.load_object(item["path"])
        if self.instance_transforms is not None:
            if not callable(self.instance_transforms):
                self.instance_transforms = instantiate(self.instance_transforms)
            data_object = self.instance_transforms(data_object)
        return {
            "data_object": data_object,
            "label": item["label"],
            "path": item["path"],
            "speaker_id": item.get("speaker_id"),
            "audio_name": item.get("audio_name"),
        }
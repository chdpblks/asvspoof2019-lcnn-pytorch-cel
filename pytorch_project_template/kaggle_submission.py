import os
import sys
import torch
import torch.nn as nn
import numpy as np
import librosa
from pathlib import Path
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader

# IMPORTANT: If you get numpy compatibility errors, run this in a separate cell first:
# !pip install numpy==1.26.4 --quiet
# Then restart the kernel and run this script again.

# Add project to path
sys.path.insert(0, '/kaggle/working/asvspoof2019-lcnn-pytorch-cel/pytorch_project_template')

from src.model.light_cnn import LightCNN_ASVspoof2019
from src.transforms.lfcc import LFCC


# ==================== Model Definition ====================
class MFM(nn.Module):
    """Max-Feature-Map: split channels on two parts and take element-wise max."""

    def __init__(self, in_channels, out_channels, type='conv'):
        super(MFM, self).__init__()
        self.out_channels = out_channels
        if type == 'conv':
            self.filter = nn.Conv2d(in_channels, 2 * out_channels, kernel_size=1, stride=1, padding=0)
        elif type == 'linear':
            self.filter = nn.Linear(in_channels, 2 * out_channels)
        else:
            raise ValueError('Unknown type: {}'.format(type))

    def forward(self, x):
        x = self.filter(x)
        a, b = torch.split(x, self.out_channels, dim=1)
        return torch.max(a, b)


# ==================== Dataset ====================
class ASVSpoofEvalDataset(Dataset):
    """Dataset for ASVspoof2019 LA eval partition."""
    
    def __init__(self, data_dir, protocol_path):
        self.data_dir = Path(data_dir)
        self.audio_dir = self.data_dir / "ASVspoof2019_LA_eval" / "flac"
        
        self.samples = []
        with open(protocol_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) <= 4:
                    continue
                speaker_id = parts[0]
                audio_name = parts[1]
                label_str = parts[4]
                label = 1 if label_str == "bonafide" else 0
                audio_path = self.audio_dir / f"{audio_name}.flac"
                
                if audio_path.exists():
                    self.samples.append({
                        "path": str(audio_path),
                        "audio_name": audio_name,
                        "speaker_id": speaker_id,
                        "label": label
                    })
        
        print(f"Loaded {len(self.samples)} samples from {protocol_path}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load audio
        audio, sr = librosa.load(sample["path"], sr=16000, mono=True)
        audio = torch.from_numpy(audio).float()
        
        return {
            "audio": audio,
            "audio_name": sample["audio_name"],
            "speaker_id": sample["speaker_id"],
            "label": sample["label"]
        }


# ==================== Transform Classes ====================
class TrimRepeat(nn.Module):
    """Makes trim of audio and repeat if too short."""
    
    def __init__(self, target_length=78400, axis=0):
        super().__init__()
        self.target_length = target_length
        self.axis = axis
    
    def forward(self, x):
        if torch.is_tensor(x):
            x = x.cpu().numpy()
        
        # 1D audio case
        if x.ndim == 1:
            x = x.squeeze()
            current_length = len(x)
            
            if current_length > self.target_length:
                x = x[:self.target_length]
            elif current_length < self.target_length:
                n_repeats = int(np.ceil(self.target_length / current_length))
                x = np.tile(x, n_repeats)[:self.target_length]
        
        x = x.astype(np.float32)
        return torch.from_numpy(x)


class Normalize1D(nn.Module):
    """Batch-version of Normalize for 1D Input."""
    
    def __init__(self, mean, std):
        super().__init__()
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)
    
    def forward(self, x):
        x = (x - self.mean) / self.std
        return x


# ==================== Collate Function ====================
def collate_fn(batch):
    """Collate function with LFCC transformation."""
    # Initialize transforms (same as in config)
    trim_repeat = TrimRepeat(target_length=78400)
    lfcc_transform = LFCC(sample_rate=16000, n_lfcc=20)
    normalize = Normalize1D(mean=[0.5], std=[0.5])
    
    audios = [item["audio"] for item in batch]
    audio_names = [item["audio_name"] for item in batch]
    speaker_ids = [item["speaker_id"] for item in batch]
    labels = [item["label"] for item in batch]
    
    # Apply transformations
    spectrograms = []
    for audio in audios:
        # Trim/repeat to target length
        audio = trim_repeat(audio)
        
        # Compute LFCC
        spec = lfcc_transform(audio.unsqueeze(0))  # Add channel dimension
        
        # Normalize
        spec = normalize(spec)
        
        spectrograms.append(spec)
    
    spectrograms = torch.stack(spectrograms)
    labels = torch.tensor(labels, dtype=torch.long)
    
    return {
        "data_object": spectrograms,
        "labels": labels,
        "audio_name": audio_names,
        "speaker_id": speaker_ids
    }


# ==================== Main Function ====================
def generate_submission():
    """Generate submission files."""
    
    # Configuration
    CHECKPOINT_PATH = "/kaggle/working/abcdef/model_best.pth"
    DATA_DIR = "/kaggle/input/datasets/awsaf49/asvpoof-2019-dataset/LA/LA"
    PROTOCOL_PATH = "/kaggle/input/datasets/awsaf49/asvpoof-2019-dataset/LA/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt"
    OUTPUT_DIR = "/kaggle/working/submission"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Using device: {DEVICE}")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load dataset
    print("Loading dataset...")
    dataset = ASVSpoofEvalDataset(DATA_DIR, PROTOCOL_PATH)
    dataloader = DataLoader(
        dataset, 
        batch_size=16,  # Same as in config
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=2
    )
    
    # Load model
    print("Loading model...")
    model = LightCNN_ASVspoof2019(num_classes=2, dropout_prob=0.1)
    
    # Load checkpoint - handle numpy 2.x compatibility issue
    import numpy as np
    print(f"NumPy version: {np.__version__}")
    
    # If numpy 2.x, we need to downgrade first
    if np.__version__.startswith('2.'):
        raise RuntimeError(
            "NumPy 2.x detected. Please downgrade numpy first by running this in a separate cell:\n"
            "!pip install numpy==1.26.4 --quiet\n"
            "Then restart the kernel and run this script again."
        )
    
    import pickle
    with open(CHECKPOINT_PATH, 'rb') as f:
        checkpoint = pickle.load(f)
    
    model.load_state_dict(checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint)
    model = model.to(DEVICE)
    model.eval()
    
    print("Running inference...")
    all_predictions = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader):
            # Move to device
            data_object = batch["data_object"].to(DEVICE)
            
            # Forward pass
            outputs = model(data_object)
            logits = outputs["logits"]
            
            # Convert to probabilities
            probs = torch.softmax(logits, dim=-1)
            bonafide_scores = probs[:, 1].cpu().numpy()  # Probability of bonafide
            
            # Collect predictions
            for i in range(len(batch["audio_name"])):
                audio_name = batch["audio_name"][i]
                speaker_id = batch["speaker_id"][i]
                label = batch["labels"][i].item()
                score = bonafide_scores[i]
                
                # Extract attack type from audio_name
                parts = audio_name.split('_')
                if len(parts) >= 4 and parts[3].startswith('A'):
                    attack_type = parts[3].split('-')[0]
                else:
                    attack_type = '-'
                
                all_predictions.append({
                    "speaker_id": speaker_id,
                    "audio_name": audio_name,
                    "label": label,
                    "attack_type": attack_type,
                    "score": score
                })
    
    # Sort by audio_name
    all_predictions.sort(key=lambda x: x["audio_name"])
    
    # Generate score_cm.txt
    print("Generating score_cm.txt...")
    score_cm_path = os.path.join(OUTPUT_DIR, "score_cm.txt")
    with open(score_cm_path, "w") as f:
        for pred in all_predictions:
            label = "bonafide" if pred["label"] == 1 else "spoof"
            f.write(f"{pred['speaker_id']}  {pred['attack_type']}  {label}  {pred['score']:.6f}\n")
    
    print(f"Saved score_cm.txt to {score_cm_path}")
    
    # Generate CSV
    print("Generating tagirustisanbirdin.csv...")
    csv_path = os.path.join(OUTPUT_DIR, "tagirustisanbirdin.csv")
    with open(csv_path, "w") as f:
        for pred in all_predictions:
            f.write(f"{pred['audio_name']},{pred['score']:.6f}\n")
    
    print(f"Saved tagirustisanbirdin.csv to {csv_path}")
    
    # Print statistics
    print(f"\nTotal predictions: {len(all_predictions)}")
    bonafide_count = sum(1 for p in all_predictions if p["label"] == 1)
    spoof_count = sum(1 for p in all_predictions if p["label"] == 0)
    print(f"Bonafide samples: {bonafide_count}")
    print(f"Spoof samples: {spoof_count}")
    print(f"Score range: [{min(p['score'] for p in all_predictions):.4f}, {max(p['score'] for p in all_predictions):.4f}]")
    
    print("\nDone!")


if __name__ == "__main__":
    generate_submission()

import torchaudio
from torch.utils.data import Dataset
import os

dataset = torchaudio.datasets.YESNO(root="./yes_no", download=True)

class YesNoDataset(Dataset):
    def __init__(self,root_dir, label_dir)->None:
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.audiofiles = os.listdir(self.path)
    def __getitem__(self, index):
        audio_name = self.audiofiles[index]
        wave_path = os.path.join(self.path, audio_name)
        waveform, sr = torchaudio.load(wave_path)
        return waveform, sr, audio_name
    def __len__(self):
        return len(self.audiofiles )
    

    



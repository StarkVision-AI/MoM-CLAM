import torch
import os
import pandas as pd
from torch.utils.data import Dataset
from tqdm.auto import tqdm # Import tqdm

class SongsDataset(Dataset):
    def __init__(self, df, mert_dir, wav2vec2_dir, label = 0, split="train"):
        self.df = df
        self.mert_dir = mert_dir
        self.wav2vec2_dir = wav2vec2_dir

        self.mert_embeddings = []
        self.wav2vec2_embeddings = []
        self.label = label
        self.label_dict = { 0 : "real", 1 : "fake" }
        loaded_filenames = []
        print(f"Preloading embeddings for {split} split...")

        for index, row in tqdm(self.df.iterrows(), total=len(self.df), desc=f"Loading {split} data to CPU"):
            filename = row['filename']
            filename_original = filename
            if filename.endswith(".mp3"):
                filename = filename[:-4]
            filename = filename + ".pt"

            mert_path = os.path.join(self.mert_dir, filename)
            wav2vec2_path = os.path.join(self.wav2vec2_dir, filename)


            try:
                mert = torch.load(mert_path, map_location=torch.device('cpu'))
                self.mert_embeddings.append(mert)


                wav2vec2 = torch.load(wav2vec2_path, map_location=torch.device('cpu'))
                self.wav2vec2_embeddings.append(wav2vec2)

                loaded_filenames.append(filename_original)
            except FileNotFoundError as e:
                print(f"\nWarning: File not found for {filename}: {e}. Skipping sample.")
                continue
            except Exception as e:
                print(e)
                print(f"\nWarning: Error loading embeddings for {filename}: {e}. Skipping sample.")
                continue


        self.df = self.df[self.df['filename'].isin(loaded_filenames)].reset_index(drop=True)
        self.loaded_filenames = loaded_filenames




    def __len__(self):
        return  len(self.loaded_filenames)

    def __getitem__(self, idx):
        mert = self.mert_embeddings[idx]
        wav2vec2 = self.wav2vec2_embeddings[idx] 


        return mert, wav2vec2, self.label 
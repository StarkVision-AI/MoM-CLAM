wav2vec2 = "m3hrdadfi/wav2vec2-base-100k-gtzan-music-genres"

from transformers import Wav2Vec2FeatureExtractor
from transformers import AutoModel
import torch
from torch import nn
import torchaudio.transforms as T   
import torchaudio
import os
from tqdm import tqdm

model = AutoModel.from_pretrained(wav2vec2, trust_remote_code=True)
processor = Wav2Vec2FeatureExtractor.from_pretrained(wav2vec2, trust_remote_code=True)

# loading the model to GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

def wav2vec2_embedding_fn(audio , duration = 90 ,  device = "cuda"):
    sr = processor.sampling_rate
    # resample to 16kHz
    audio , sr_ = torchaudio.load(audio)
    if sr_ != sr:
        resampler = T.Resample(sr_, sr)
        audio = resampler(audio)

    # convert to mono
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)

    # pad or truncate to 90 seconds
    if audio.shape[1] > sr * duration:
        audio = audio[:, :sr * duration]
    elif audio.shape[1] < sr * duration:
        pad = torch.zeros(1, sr * duration - audio.shape[1])
        audio = torch.cat([audio, pad], dim=1)
    # get the embeddings

    audio = audio.to(device)

    with torch.no_grad():
        embedding = model(**processor(audio.squeeze(), sampling_rate=sr, return_tensors="pt").to(device) , output_hidden_states=True)

    layer_embeddings = torch.stack(embedding.hidden_states).squeeze()

    return layer_embeddings.mean(1)


#Take argument as mp3 folder in and out
import argparse
import glob
import pandas as pd
import numpy as np
import json
import shutil
import random
import time
import sys


arg = argparse.ArgumentParser() 
arg.add_argument("--input", type=str, help="input folder")
arg.add_argument("--output", type=str, help="output folder")


def main(): 
    args = arg.parse_args()
    input_folder = args.input
    output_folder = args.output

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    mp3_files = glob.glob(os.path.join(input_folder, "*.mp3"))

    if len(mp3_files) == 0:
        print("No mp3 files found in the input folder.")
        sys.exit(1)

    # Create a list to store the embeddings

    for i, mp3_file in enumerate(tqdm(mp3_files)):
        try:
            # Get the file name without the extension
            file_name = os.path.splitext(os.path.basename(mp3_file))[0]
            # Create the output file path
            output_file = os.path.join(output_folder, f"{file_name}.pt")
            # Check if the output file already exists
            if os.path.exists(output_file):
                print(f"Output file {output_file} already exists. Skipping.")
                continue
            # Get the embeddings
            embedding = wav2vec2_embedding_fn(mp3_file)
            # Save the embeddings to a file
            torch.save(embedding, output_file)
        except Exception as e:
            print(f"Error processing {mp3_file}: {e}")
            continue

if __name__ == "__main__":
    main()

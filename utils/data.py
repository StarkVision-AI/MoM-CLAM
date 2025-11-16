import pandas as pd
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from utils import dataset
import torch

def remove_filename_not_found(df, folder):
    filenames_in_folder = os.listdir(folder)
    df_copy = df.copy()
    
    # Create a set of filenames (without .pt) for faster lookup
    filenames_set = set()
    for fname in filenames_in_folder:
        if fname.endswith('.pt'):
            filenames_set.add(fname[:-3]) # Remove .pt extension
    
    indices_to_drop = []
    for index, row in tqdm(df_copy.iterrows(), total=df_copy.shape[0], desc=f"Filtering {folder}"):
        filename = row['filename']
        # Remove common audio extensions for comparison
        if filename.endswith(('.mp3', '.wav')):
            filename_base = filename[:-4]
        else:
            filename_base = filename
        
        if filename_base + '.pt' not in filenames_in_folder: # Check if the .pt file exists
            indices_to_drop.append(index)
            
    df_copy.drop(indices_to_drop, inplace=True)
    return df_copy


def prepare_datasets(real_df_path1, real_df_path2, fake_df_path,
                     real_mert_folder1, real_wav2vec2_folder1,
                     real_mert_folder2, real_wav2vec2_folder2,
                     fake_mert_folder, fake_wav2vec2_folder):

    # Load dataframes
    real_df_1 = pd.read_csv(real_df_path1)
    real_df_2 = pd.read_csv(real_df_path2)
    fake_df = pd.read_csv(fake_df_path)

    print(f"Initial lengths: real_df_1={len(real_df_1)}, real_df_2={len(real_df_2)}, fake_df={len(fake_df)}")

    # Remove entries where files are missing
    real_df_1 = remove_filename_not_found(real_df_1, real_mert_folder1)
    real_df_2 = remove_filename_not_found(real_df_2, real_mert_folder2) # Note: assumes mert and wav2vec2 have same files
    fake_df = remove_filename_not_found(fake_df, fake_mert_folder)
    
    print(f"Lengths after filtering: real_df_1={len(real_df_1)}, real_df_2={len(real_df_2)}, fake_df={len(fake_df)}")


    real_df = pd.concat([real_df_1, real_df_2]).reset_index(drop=True)
    print(f"Combined real_df length: {len(real_df)}")

    # Split fake dataset based on model_name
    fake_df_test = fake_df[fake_df["model_name"].isin(["riffusion", "suno_3", "AI_COVERS", 'suno_4', 'Yue'])]
    fake_df_train = fake_df[~fake_df["model_name"].isin(["riffusion", "suno_3", "AI_COVERS", "suno_4", "Yue"])]

    print(f"Fake train length: {len(fake_df_train)}, Fake test length: {len(fake_df_test)}")

    ratio = len(fake_df_train) / (len(fake_df_test) + len(fake_df_train))
    real_df_train, real_df_test = train_test_split(real_df, test_size=1 - ratio, random_state=42)

    print(f"Real train length: {len(real_df_train)}, Real test length: {len(real_df_test)}")

    # Create datasets
    train_real_dataset = dataset.SongsDataset(real_df_train, real_mert_folder1, real_wav2vec2_folder1, label=0)
    train_fake_dataset = dataset.SongsDataset(fake_df_train, fake_mert_folder, fake_wav2vec2_folder, label=1)

    # For test datasets, we need to handle the different source folders correctly.
    # Assuming real_mert_folder1 and real_wav2vec2_folder1 contain the real test data.
    test_real_dataset = dataset.SongsDataset(real_df_test, real_mert_folder1, real_wav2vec2_folder1, label=0, split="test")
    test_fake_dataset = dataset.SongsDataset(fake_df_test, fake_mert_folder, fake_wav2vec2_folder, label=1, split="test")

    train_dataset = torch.utils.data.ConcatDataset([train_real_dataset, train_fake_dataset])
    test_dataset = torch.utils.data.ConcatDataset([test_real_dataset, test_fake_dataset])

    print(f"Combined train dataset length: {len(train_dataset)}")
    print(f"Combined test dataset length: {len(test_dataset)}")

    return train_dataset, test_dataset
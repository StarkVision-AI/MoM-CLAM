# %%
from utils import dataset
from models.clam import CLAM
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import os


# %%
#Real Dataset



real_df_1 = "real_songs_2.csv"
real_df_1 = pd.read_csv(real_df_1)
real_df_2 = "real_yt_covers.csv"
real_df_2 = pd.read_csv(real_df_2)


real_mert_1 = "real_songs_mert"
real_mert_2 = "yt_covers_mert"

real_wav2vec2_1 = "real_songs_wav2vec2"
real_wav2vec2_2 = "yt_covers_wav2vec2"

print(len(real_df_1))
print(len(os.listdir(real_mert_1)))
print(len(os.listdir(real_wav2vec2_1)))

print(len(real_df_2))
print(len(os.listdir(real_mert_2)))
print(len(os.listdir(real_wav2vec2_2)))

# %%
import pandas as pd
fake_df = "ai_generated_music_metadata.csv"
fake_df = pd.read_csv(fake_df)

# %%
fake_df.head()

# %%
#Fake Dataset


fake_df = "ai_generated_music_metadata.csv"
fake_df = pd.read_csv(fake_df)
fake_mert = "ai_generated_music_mert"
fake_wav2vec2 = "ai_generated_music_wav2vec2"


print(len(fake_df))
print(len(os.listdir(fake_mert)))
print(len(os.listdir(fake_wav2vec2)))



# %%
from tqdm import tqdm

def remove_filename(df, folder):
    filenames = os.listdir(folder)
    df_copy = df.copy()
    for index, row in tqdm(df_copy.iterrows(), total=df_copy.shape[0]):
        filename = row['filename']
        if filename.endswith('.mp3'):
            filename = filename[:-4]
        if filename.endswith('.wav'):
            filename = filename[:-4]

        if filename + '.pt' not in filenames:
            df_copy.drop(index, inplace=True)
    return df_copy


real_df_1 = remove_filename(real_df_1, real_mert_1)
real_df_2 = remove_filename(real_df_2, real_mert_2)
fake_df = remove_filename(fake_df, fake_mert)
print(len(real_df_1))
print(len(real_df_2))
print(len(fake_df))

# %%
print(len(real_df_1))
print(len(real_df_2))

# %%
real_df = pd.concat([real_df_1, real_df_2])
print(len(real_df))

# %%
fake_df["model_name"].value_counts()

# %%
from sklearn.model_selection import train_test_split


fake_df_test = fake_df[fake_df["model_name"].isin(["riffusion", "suno_3", "AI_COVERS"  , 'suno_4' , 'Yue'])]
print(len(fake_df_test))
fake_df_train = fake_df[~fake_df["model_name"].isin(["riffusion", "suno_3", "AI_COVERS" , "suno_4" , "Yue"] )]
print(len(fake_df_train))

ratio = len(fake_df_train) / ( len(fake_df_test) + len(fake_df_train)  )
print(ratio)

real_df = pd.concat([real_df_1, real_df_2])

real_df_train, real_df_test = train_test_split(real_df, test_size= 1 - ratio, random_state=42)
print(len(real_df_train))
print(len(real_df_test))

# %%
from utils import dataset

train_real_dataset = dataset.SongsDataset(real_df_train, real_mert_1, real_wav2vec2_1, label = 0)
train_fake_dataset = dataset.SongsDataset(fake_df_train, fake_mert, fake_wav2vec2, label = 1)

test_real_dataset = dataset.SongsDataset(real_df_test, real_mert_1, real_wav2vec2_1, label = 0 , split = "test")
test_fake_dataset = dataset.SongsDataset(fake_df_test, fake_mert, fake_wav2vec2, label = 1 , split = "test")


# %%
print(len(train_real_dataset))
print(len(train_fake_dataset))
print(len(test_real_dataset))
print(len(test_fake_dataset))


# %%
train_dataset = torch.utils.data.ConcatDataset([train_real_dataset, train_fake_dataset])
test_dataset = torch.utils.data.ConcatDataset([test_real_dataset, test_fake_dataset])
print(len(train_dataset))
print(len(test_dataset))

# %%


from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.optim import Adam , AdamW
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import warnings
from sklearn.metrics import roc_curve
import numpy as np

def compute_eer(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    # Find the point where FPR = FNRbest_model_l1_loss.pth
    eer_threshold_index = np.nanargmin(np.absolute((fnr - fpr)))
    eer = (fpr[eer_threshold_index] + fnr[eer_threshold_index]) / 2
    return eer


val_split = 0.2
train_size = int((1 - val_split) * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size] , generator=torch.Generator().manual_seed(42))
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
in_channel1 = 13
in_channel2 = 13
embed_dim1 = 768
embed_dim2 = 768
best_acc = 0
best_f1 = 0

model = CLAM(in_channel1 , in_channel2 , embed_dim1 , embed_dim2).to(device)

optimizer = AdamW(model.parameters(), lr=1e-4)

classification_criterion = nn.BCEWithLogitsLoss() # For fake/real classification
margin = 0.2 # You might need to tune this hyperparameter
alignment_criterion = nn.TripletMarginLoss(margin=margin)
save_name = f"best_model_triplet_loss_margin_{margin}.pth" # Update save name
save_folder = "model_wts"
# --- Hyperparameters ---
epochs = 50
alignment_loss_weight = 0.5 # Weight factor for the alignment loss 
print("Starting Training...")




model.load_state_dict(torch.load(os.path.join(save_folder, f"{save_name}")))
model.eval()
test_loss_total = 0
test_loss_cls = 0
test_loss_align = 0
test_preds_all = []
test_labels_all = []
test_scores_all = []    


with torch.no_grad():
    for batch in tqdm(test_loader, desc="Testing", unit="batch"):
        embed1, embed2, labels = batch
        embed1 = embed1.to(device)
        embed2 = embed2.to(device)
        labels = labels.float().to(device).view(-1)

        # --- Forward pass (testing) ---
        outputs , real_emb, fake_emb  = model.forward_training(embed1, embed2)
        outputs = outputs.view(-1)

        # --- Calculate Classification Loss (Testing) ---
        classification_loss = classification_criterion(outputs, labels)

        real_indices = (labels == 1).nonzero(as_tuple=True)[0]
        alignment_loss = torch.tensor(0.0).to(device)
        if real_indices.nelement() > 0:
            real_emb_filtered = real_emb[real_indices]
            fake_emb_filtered = fake_emb[real_indices]
            
            num_real_samples = real_emb_filtered.size(0)
            if num_real_samples > 1:
                triplet_losses = []
                for i in range(num_real_samples):
                    anchor = real_emb_filtered[i]
                    positive = fake_emb_filtered[i]
                    negative_indices = [j for j in range(num_real_samples) if j != i]
                    negatives = fake_emb_filtered[negative_indices]

                    for neg in negatives:
                        loss = alignment_criterion(anchor.unsqueeze(0), positive.unsqueeze(0), neg.unsqueeze(0))
                        triplet_losses.append(loss)

                if triplet_losses:
                    alignment_loss = torch.mean(torch.stack(triplet_losses))
                else:
                    alignment_loss = torch.tensor(0.0).to(device)

        total_loss = classification_loss + alignment_loss_weight * alignment_loss

        test_loss_total += total_loss.item()
        test_loss_cls += classification_loss.item()
        test_loss_align += alignment_loss.item()

        preds = torch.sigmoid(outputs).detach().round()
        test_preds_all.extend(preds.cpu().numpy())
        test_labels_all.extend(labels.cpu().numpy())
        test_scores_all.extend(torch.sigmoid(outputs).cpu().numpy())

    num_batches_test = len(test_loader)
    avg_test_loss_total = test_loss_total / num_batches_test
    avg_test_loss_cls = test_loss_cls / num_batches_test    
    avg_test_loss_align = test_loss_align / num_batches_test
    test_acc = accuracy_score(test_labels_all, test_preds_all)  
    test_f1 = f1_score(test_labels_all, test_preds_all, average='binary', zero_division=0)
    test_recall = recall_score(test_labels_all, test_preds_all, average='binary', zero_division=0)
    test_precision = precision_score(test_labels_all, test_preds_all, average='binary', zero_division=0)
    eer = compute_eer(test_labels_all, test_scores_all)  # Calculate EER using the raw scores
    print(f"Test Total Loss: {avg_test_loss_total:.4f} | CLS Loss: {avg_test_loss_cls:.4f} | Align Loss (raw): {avg_test_loss_align:.4f}")
    print(f"Test Acc: {test_acc:.4f} | F1: {test_f1:.4f} | Recall: {test_recall:.4f} | Precision: {test_precision:.4f}")
    print(f"EER: {eer:.4f}")

# %%


from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.optim import Adam , AdamW
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import warnings
from sklearn.metrics import roc_curve
import numpy as np

def compute_eer(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    # Find the point where FPR = FNRbest_model_l1_loss.pth
    eer_threshold_index = np.nanargmin(np.absolute((fnr - fpr)))
    eer = (fpr[eer_threshold_index] + fnr[eer_threshold_index]) / 2
    return eer


# train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)

val_split = 0.2
train_size = int((1 - val_split) * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size] , generator=torch.Generator().manual_seed(42))
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
in_channel1 = 13
in_channel2 = 13
embed_dim1 = 768
embed_dim2 = 768
best_acc = 0
best_f1 = 0

model = CLAM(in_channel1 , in_channel2 , embed_dim1 , embed_dim2).to(device)

optimizer = AdamW(model.parameters(), lr=1e-4)

classification_criterion = nn.BCEWithLogitsLoss() # For fake/real classification
margin = 0.2 # You might need to tune this hyperparameter
alignment_criterion = nn.TripletMarginLoss(margin=margin)
save_name = f"best_model_triplet_loss_margin_{margin}.pth" # Update save name
save_folder = "model_wts"
# --- Hyperparameters ---
epochs = 50
alignment_loss_weight = 0.5 # Weight factor for the alignment loss 
print("Starting Training...")


for epoch in range(epochs):

    model.train()
    train_loss_total = 0
    train_loss_cls = 0
    train_loss_align = 0
    train_preds_all = []
    train_labels_all = []

    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} Training", unit="batch"):
        optimizer.zero_grad()

        embed1, embed2, labels = batch
        embed1 = embed1.to(device)
        embed2 = embed2.to(device)
        labels = labels.float().to(device).view(-1)

        # --- Forward pass ---
        outputs , real_emb, fake_emb  = model.forward_training(embed1, embed2)
        outputs = outputs.view(-1) 

        # --- Calculate Classification Loss (for all samples) ---
        classification_loss = classification_criterion(outputs, labels)

 
        real_indices = (labels == 0).nonzero(as_tuple=True)[0]

        alignment_loss = torch.tensor(0.0).to(device) # Initialize alignment loss for this batch

        if real_indices.nelement() > 0: # Check if there are any real samples in the batch
            # 2. Select the embeddings corresponding to real samples
            real_emb_filtered = real_emb[real_indices] # These will be anchors
            fake_emb_filtered = fake_emb[real_indices] # These will be positives

            num_real_samples = real_emb_filtered.size(0)

            # We need at least 2 real samples in the batch to form triplets with negatives
            if num_real_samples > 1:
                triplet_losses = []
                # In-batch mining: Iterate through each real sample as anchor/positive
                for i in range(num_real_samples):
                    anchor = real_emb_filtered[i]
                    positive = fake_emb_filtered[i]

                    # Select all *other* fake embeddings from real samples in the batch as negatives
                    # We need to ensure the negative is not the positive sample itself
                    negative_indices = [j for j in range(num_real_samples) if j != i]
                    negatives = fake_emb_filtered[negative_indices]

                    for neg in negatives:

                        loss = alignment_criterion(anchor.unsqueeze(0), positive.unsqueeze(0), neg.unsqueeze(0))
                        triplet_losses.append(loss)

                if triplet_losses: # Ensure the list is not empty
                    alignment_loss = torch.mean(torch.stack(triplet_losses)) # Average the triplet losses
                else:
                    alignment_loss = torch.tensor(0.0).to(device) # No valid triplets formed

        # --- Combine Losses ---
        total_loss = classification_loss + alignment_loss_weight * alignment_loss
        total_loss.backward()
        optimizer.step()

        # --- Accumulate metrics and losses for reporting ---
        train_loss_total += total_loss.item()
        train_loss_cls += classification_loss.item()
        train_loss_align += alignment_loss.item() # Note: this is the raw alignment loss before weighting


        preds = torch.sigmoid(outputs).detach().round() # Get predictions (0 or 1)
        train_preds_all.extend(preds.cpu().numpy())
        train_labels_all.extend(labels.cpu().numpy())

    # --- Calculate Epoch Metrics (Training) ---
    num_batches = len(train_loader)
    avg_train_loss_total = train_loss_total / num_batches
    avg_train_loss_cls = train_loss_cls / num_batches
    avg_train_loss_align = train_loss_align / num_batches # Average raw alignment loss across batches where it was calculated

    train_acc = accuracy_score(train_labels_all, train_preds_all)
    train_f1 = f1_score(train_labels_all, train_preds_all, average='binary', zero_division=0)
    train_recall = recall_score(train_labels_all, train_preds_all, average='binary', zero_division=0)
    train_precision = precision_score(train_labels_all, train_preds_all, average='binary', zero_division=0)

    print(f"Epoch {epoch+1}/{epochs}")
    print(f"  Train Total Loss: {avg_train_loss_total:.4f} | CLS Loss: {avg_train_loss_cls:.4f} | Align Loss (raw): {avg_train_loss_align:.4f}")
    print(f"  Train Acc: {train_acc:.4f} | F1: {train_f1:.4f} | Recall: {train_recall:.4f} | Precision: {train_precision:.4f}")


    # --- Validation ---
    model.eval()
    val_loss_total = 0
    val_loss_cls = 0
    val_loss_align = 0 
    val_preds_all = []
    val_labels_all = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} Validation", unit="batch"):
            embed1, embed2, labels = batch
            embed1 = embed1.to(device)
            embed2 = embed2.to(device)
            labels = labels.float().to(device).view(-1)

            # --- Forward pass (validation) ---
            outputs , real_emb, fake_emb  = model.forward_training(embed1, embed2)
            outputs = outputs.view(-1)

            classification_loss = classification_criterion(outputs, labels)

            real_indices = (labels == 0).nonzero(as_tuple=True)[0]
            alignment_loss = torch.tensor(0.0).to(device) # Initialize alignment loss

            if real_indices.nelement() > 0:
                real_emb_filtered = real_emb[real_indices]
                fake_emb_filtered = fake_emb[real_indices]
                num_real_samples = real_emb_filtered.size(0)

                if num_real_samples > 1:
                    triplet_losses = []
                    for i in range(num_real_samples):
                        anchor = real_emb_filtered[i]
                        positive = fake_emb_filtered[i]
                        negative_indices = [j for j in range(num_real_samples) if j != i]
                        negatives = fake_emb_filtered[negative_indices]

                        for neg in negatives:
                            loss = alignment_criterion(anchor.unsqueeze(0), positive.unsqueeze(0), neg.unsqueeze(0))
                            triplet_losses.append(loss)

                    if triplet_losses:
                        alignment_loss = torch.mean(torch.stack(triplet_losses))
                    else:
                        alignment_loss = torch.tensor(0.0).to(device)


            total_loss = classification_loss + alignment_loss_weight * alignment_loss
            val_loss_total += total_loss.item()
            val_loss_cls += classification_loss.item()
            val_loss_align += alignment_loss.item()

            preds = torch.sigmoid(outputs).detach().round()
            val_preds_all.extend(preds.cpu().numpy())
            val_labels_all.extend(labels.cpu().numpy())

    num_batches_val = len(val_loader)
    avg_val_loss_total = val_loss_total / num_batches_val
    avg_val_loss_cls = val_loss_cls / num_batches_val
    avg_val_loss_align = val_loss_align / num_batches_val

    val_acc = accuracy_score(val_labels_all, val_preds_all)
    val_f1 = f1_score(val_labels_all, val_preds_all, average='binary', zero_division=0)
    val_recall = recall_score(val_labels_all, val_preds_all, average='binary', zero_division=0)
    val_precision = precision_score(val_labels_all, val_preds_all, average='binary', zero_division=0)

    print(f"  Val Total Loss: {avg_val_loss_total:.4f} | CLS Loss: {avg_val_loss_cls:.4f} | Align Loss (raw): {avg_val_loss_align:.4f}")
    print(f"  Val Acc: {val_acc:.4f} | F1: {val_f1:.4f} | Recall: {val_recall:.4f} | Precision: {val_precision:.4f}")

    current_val_metric = val_acc 

    #Best F1
    if val_f1 > best_f1:
        best_f1 = val_f1
        print(f"  New Best F1: {best_f1:.4f}")
        torch.save(model.state_dict(), os.path.join(save_folder, save_name))
    else:
        print(f"  Model not saved. Best F1 so far: {best_f1:.4f}")



print("Training Finished.")
print(f"Best Validation Accuracy achieved: {best_acc:.4f}")




# %%
#On test set
model.load_state_dict(torch.load(os.path.join(save_folder, f"{save_name}")))
model.eval()
test_loss_total = 0
test_loss_cls = 0
test_loss_align = 0
test_preds_all = []
test_labels_all = []
test_scores_all = []    


with torch.no_grad():
    for batch in tqdm(test_loader, desc="Testing", unit="batch"):
        embed1, embed2, labels = batch
        embed1 = embed1.to(device)
        embed2 = embed2.to(device)
        labels = labels.float().to(device).view(-1)

        # --- Forward pass (testing) ---
        outputs , real_emb, fake_emb  = model.forward_training(embed1, embed2)
        outputs = outputs.view(-1)

        # --- Calculate Classification Loss (Testing) ---
        classification_loss = classification_criterion(outputs, labels)

        real_indices = (labels == 1).nonzero(as_tuple=True)[0]
        alignment_loss = torch.tensor(0.0).to(device)
        if real_indices.nelement() > 0:
            real_emb_filtered = real_emb[real_indices]
            fake_emb_filtered = fake_emb[real_indices]
            
            num_real_samples = real_emb_filtered.size(0)
            if num_real_samples > 1:
                triplet_losses = []
                for i in range(num_real_samples):
                    anchor = real_emb_filtered[i]
                    positive = fake_emb_filtered[i]
                    negative_indices = [j for j in range(num_real_samples) if j != i]
                    negatives = fake_emb_filtered[negative_indices]

                    for neg in negatives:
                        loss = alignment_criterion(anchor.unsqueeze(0), positive.unsqueeze(0), neg.unsqueeze(0))
                        triplet_losses.append(loss)

                if triplet_losses:
                    alignment_loss = torch.mean(torch.stack(triplet_losses))
                else:
                    alignment_loss = torch.tensor(0.0).to(device)

        total_loss = classification_loss + alignment_loss_weight * alignment_loss

        test_loss_total += total_loss.item()
        test_loss_cls += classification_loss.item()
        test_loss_align += alignment_loss.item()

        preds = torch.sigmoid(outputs).detach().round()
        test_preds_all.extend(preds.cpu().numpy())
        test_labels_all.extend(labels.cpu().numpy())
        test_scores_all.extend(torch.sigmoid(outputs).cpu().numpy())

    num_batches_test = len(test_loader)
    avg_test_loss_total = test_loss_total / num_batches_test
    avg_test_loss_cls = test_loss_cls / num_batches_test    
    avg_test_loss_align = test_loss_align / num_batches_test
    test_acc = accuracy_score(test_labels_all, test_preds_all)  
    test_f1 = f1_score(test_labels_all, test_preds_all, average='binary', zero_division=0)
    test_recall = recall_score(test_labels_all, test_preds_all, average='binary', zero_division=0)
    test_precision = precision_score(test_labels_all, test_preds_all, average='binary', zero_division=0)
    eer = compute_eer(test_labels_all, test_scores_all)  # Calculate EER using the raw scores
    print(f"Test Total Loss: {avg_test_loss_total:.4f} | CLS Loss: {avg_test_loss_cls:.4f} | Align Loss (raw): {avg_test_loss_align:.4f}")
    print(f"Test Acc: {test_acc:.4f} | F1: {test_f1:.4f} | Recall: {test_recall:.4f} | Precision: {test_precision:.4f}")
    print(f"EER: {eer:.4f}")

# %%
# Testing: 100%|██████████| 2139/2139 [00:03<00:00, 595.20batch/s]
# Test Total Loss: 0.4560 | CLS Loss: 0.4164 | Align Loss (raw): 0.0793
# Test Acc: 0.9185 | F1: 0.9114 | Recall: 0.8414 | Precision: 0.9942
# EER: 0.0625



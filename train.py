import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm
import warnings
import os
from sklearn.metrics import roc_curve
import numpy as np

from models.clam import CLAM
from utils.data import prepare_datasets # Import the data preparation function

def compute_eer(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    eer_threshold_index = np.nanargmin(np.absolute((fnr - fpr)))
    eer = (fpr[eer_threshold_index] + fnr[eer_threshold_index]) / 2
    return eer

def train_model(train_dataset, test_dataset, epochs=50, batch_size=16, learning_rate=1e-4, 
                alignment_loss_weight=0.5, margin=0.2, save_folder="model_wts"):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Ensure save folder exists
    os.makedirs(save_folder, exist_ok=True)

    val_split = 0.2
    train_size = int((1 - val_split) * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False) 

    in_channel1 = 13 
    in_channel2 = 13
    embed_dim1 = 768 
    embed_dim2 = 768 


    model = CLAM(in_channel1=in_channel1, in_channel2=in_channel2, embed_dim1=embed_dim1, embed_dim2=embed_dim2).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    classification_criterion = nn.BCEWithLogitsLoss()
    alignment_criterion = nn.TripletMarginLoss(margin=margin)

    save_name = f"best_model_triplet_loss_margin_{margin}.pth"
    best_f1 = 0.0

    print("Starting Training...")
    warnings.filterwarnings("ignore") # Suppress warnings

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

            outputs, real_emb, fake_emb = model.forward_training(embed1, embed2)
            outputs = outputs.view(-1)

            classification_loss = classification_criterion(outputs, labels)


            real_indices = (labels == 0).nonzero(as_tuple=True)[0] 
            alignment_loss = torch.tensor(0.0).to(device)

            if real_indices.nelement() > 0:
                real_emb_filtered = real_emb[real_indices]
                fake_emb_filtered = fake_emb[real_indices]
                num_real_samples = real_emb_filtered.size(0)

                if num_real_samples > 1:
                    triplet_losses = []
                    for i in range(num_real_samples):
                        anchor = real_emb_filtered[i]
                        positive = fake_emb_filtered[i] # Positive is the fake version of the same real song

                        negative_indices = [j for j in range(num_real_samples) if j != i]
                        negatives = fake_emb_filtered[negative_indices] # Negatives are fake versions of other real songs

                        for neg in negatives:
                            loss = alignment_criterion(anchor.unsqueeze(0), positive.unsqueeze(0), neg.unsqueeze(0))
                            triplet_losses.append(loss)

                    if triplet_losses:
                        alignment_loss = torch.mean(torch.stack(triplet_losses))
                    else:
                        alignment_loss = torch.tensor(0.0).to(device)

            total_loss = classification_loss + alignment_loss_weight * alignment_loss
            total_loss.backward()
            optimizer.step()

            train_loss_total += total_loss.item()
            train_loss_cls += classification_loss.item()
            train_loss_align += alignment_loss.item()

            preds = torch.sigmoid(outputs).detach().round()
            train_preds_all.extend(preds.cpu().numpy())
            train_labels_all.extend(labels.cpu().numpy())

        num_batches = len(train_loader)
        avg_train_loss_total = train_loss_total / num_batches
        avg_train_loss_cls = train_loss_cls / num_batches
        avg_train_loss_align = train_loss_align / num_batches

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

                outputs, real_emb, fake_emb = model.forward_training(embed1, embed2)
                outputs = outputs.view(-1)

                classification_loss = classification_criterion(outputs, labels)

                real_indices = (labels == 0).nonzero(as_tuple=True)[0]
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

        # Save best model based on F1-score
        if val_f1 > best_f1:
            best_f1 = val_f1
            print(f"  New Best F1: {best_f1:.4f}")
            torch.save(model.state_dict(), os.path.join(save_folder, save_name))
        else:
            print(f"  Model not saved. Best F1 so far: {best_f1:.4f}")

    print("Training Finished.")
    print(f"Best Validation F1-score achieved: {best_f1:.4f}")

    # Optionally, return the trained model for further use
    return model, best_f1


if __name__ == "__main__":
    # Define your paths here
    real_df_path1 = "real_songs_2.csv"
    real_df_path2 = "real_yt_covers.csv"
    fake_df_path = "ai_generated_music_metadata.csv"

    real_mert_folder1 = "real_songs_mert"
    real_wav2vec2_folder1 = "real_songs_wav2vec2"

    real_mert_folder2 = "yt_covers_mert"
    real_wav2vec2_folder2 = "yt_covers_wav2vec2"

    fake_mert_folder = "ai_generated_music_mert"
    fake_wav2vec2_folder = "ai_generated_music_wav2vec2"

    train_dataset, test_dataset = prepare_datasets(
        real_df_path1, real_df_path2, fake_df_path,
        real_mert_folder1, real_wav2vec2_folder1,
        real_mert_folder2, real_wav2vec2_folder2, # Pass the second real folders as well
        fake_mert_folder, fake_wav2vec2_folder
    )

    # Train the model
    trained_model, best_f1_score = train_model(train_dataset, test_dataset)
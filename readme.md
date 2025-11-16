# CLAM Model Training Guide

This README provides comprehensive instructions for setting up and training the CLAM model.

### 1. Dataset Acquisition and Organization

The first step is to gather and organize your audio data.

* **Download the Dataset:** All shareable data and links to publicly downloadable content for this project can be found on Hugging Face:
    [https://huggingface.co/datasets/anonymous2212/MoM-CLAM-dataset](https://huggingface.co/datasets/anonymous2212/MoM-CLAM-dataset)
* **Organize Audio Files:** After downloading, ensure all audio files are placed into their corresponding folders. The `csv` metadata file provides the necessary mapping for this organization.

### 2. Feature Embedding Extraction

Once your dataset is prepared, you'll need to extract features using the provided scripts.

* **Run Extraction Scripts:**
    * Execute `extractors/mertextraction.py`
    * Execute `extractors/wave2vec2extract.py`
* **Save Embeddings:**
    * For AI-generated music, save the MERT embeddings to `ai_generated_music_mert/` and the Wav2Vec2 embeddings to `ai_generated_music_wav2vec2/`.
    * Apply the same saving conventions for "real" music, creating similar directories (e.g., `real_music_mert/`).
* **Important:** Refer to the specific instructions within the `extractors/` files, (arguments) for detailed usage and configuration of these scripts.

### 3. Model Training

With features extracted, you're ready to train the CLAM model.

* **Initiate Training:** Run one of the following files to begin the training process and obtain the final results:
    * `train_triplet_loss.ipynb` (Jupyter Notebook for interactive training)
    * `triplet_train.py` (Python script for command-line execution)
# Instructions for running CLAM

- Organise and download all the audio files in the respective folders as shown in the csv metadeta file.
- Dataset - https://huggingface.co/datasets/anonymous2212/MoM-CLAM-dataset, it contains all the shareable data along with links for other downloadable public content.
- run "extractors/mertextraction.py" and "extractors/wave2vec2extract.py" and save embeddings in - "ai_generated_music_mert", "ai_generated_music_wav2vec2" and similar for real music using the instructions provided in the extractors file.
- then run train_triplet_loss.ipynb or triplet_train.py to achieve final results.
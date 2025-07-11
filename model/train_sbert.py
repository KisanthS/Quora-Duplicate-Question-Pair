'''# ✅ Step 1: Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# ✅ Step 2: Install Required Package
!pip install -q sentence-transformers

# ✅ Step 3: Disable wandb logging
import os
os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_DISABLED"] = "true"

# ✅ Step 4: Import required libraries
import pandas as pd
import math
import numpy as np
import re
from sentence_transformers import SentenceTransformer, InputExample, losses, util
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm.notebook import tqdm  # ✅ For progress bar

# ✅ Step 5: Load and clean dataset
def clean(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text.strip()

csv_path = '/content/drive/MyDrive/quora_data/quora_train.csv'
df = pd.read_csv(csv_path)
df.dropna(subset=['question1', 'question2', 'is_duplicate'], inplace=True)
df['question1'] = df['question1'].astype(str).apply(clean)
df['question2'] = df['question2'].astype(str).apply(clean)
df['is_duplicate'] = df['is_duplicate'].astype(float)

# ✅ Step 6: Stratified split into train/dev sets (90/10)
train_df, dev_df = train_test_split(
    df, test_size=0.1, stratify=df['is_duplicate'], random_state=42
)
train_samples = [InputExample(texts=[q1, q2], label=label)
                 for q1, q2, label in zip(train_df.question1, train_df.question2, train_df.is_duplicate)]
dev_samples = list(zip(dev_df.question1.tolist(), dev_df.question2.tolist(), dev_df.is_duplicate.tolist()))

# ✅ Step 7: Load strong SBERT model and prepare training
model = SentenceTransformer('all-MiniLM-L6-v2')
train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=32)
train_loss = losses.CosineSimilarityLoss(model=model)

# ✅ Step 8: Define training parameters
output_path = '/content/drive/MyDrive/quora_model/sbert_quora_model_best'
os.makedirs(output_path, exist_ok=True)

num_epochs = 10
warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)
best_accuracy = 0
best_threshold = 0
epochs_no_improve = 0
patience = 3

# ✅ Step 9: Training loop with evaluation + progress bar
for epoch in range(num_epochs):
    print(f"\n🚀 Epoch {epoch + 1}/{num_epochs} training...")

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=1,
        warmup_steps=warmup_steps,
        show_progress_bar=True,
        use_amp=True
    )

    # ✅ Evaluate after each epoch
    pred_scores = []
    true_labels = []

    for q1, q2, label in tqdm(dev_samples, desc="🔍 Evaluating on Dev Set"):
        emb1 = model.encode(q1, convert_to_tensor=True, show_progress_bar=False)
        emb2 = model.encode(q2, convert_to_tensor=True, show_progress_bar=False)
        score = float(util.cos_sim(emb1, emb2))
        pred_scores.append(score)
        true_labels.append(int(label))

    # ✅ Threshold tuning
    best_epoch_threshold = 0
    best_epoch_acc = 0
    for threshold in np.arange(0.6, 0.9, 0.005):
        pred_labels = [1 if s > threshold else 0 for s in pred_scores]
        acc = accuracy_score(true_labels, pred_labels) * 100
        if acc > best_epoch_acc:
            best_epoch_acc = acc
            best_epoch_threshold = threshold

    print(f"✅ Accuracy after Epoch {epoch + 1}: {best_epoch_acc:.2f}% at threshold {best_epoch_threshold:.3f}")

    # ✅ Save best model
    if best_epoch_acc > best_accuracy:
        best_accuracy = best_epoch_acc
        best_threshold = best_epoch_threshold
        model.save(output_path)
        with open(os.path.join(output_path, 'best_threshold.txt'), 'w') as f:
            f.write(str(best_threshold))
        print("✅ Best model updated and saved.")
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        print(f"⚠️ No improvement. Early stop count: {epochs_no_improve}/{patience}")
        if epochs_no_improve >= patience:
            print("\n🛑 Early stopping triggered. Ending training.")
            break

# ✅ Final report
print(f"\n📊 Best Accuracy: {best_accuracy:.2f}%")
print(f"🎯 Best Threshold: {best_threshold:.3f}")
print(f"💾 Best Model Saved at: {output_path}")'''





'''Local training'''
# ✅ Step 1: Install required package if not already installed
# pip install sentence-transformers

# ✅ Step 2: Import required libraries
import os
import pandas as pd
import math
import numpy as np
import re
from sentence_transformers import SentenceTransformer, InputExample, losses, util
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ✅ Step 3: Load and clean dataset
def clean(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text.strip()

csv_path = os.path.join("data", "quora_train.csv")
df = pd.read_csv(csv_path)
df.dropna(subset=['question1', 'question2', 'is_duplicate'], inplace=True)
df['question1'] = df['question1'].astype(str).apply(clean)
df['question2'] = df['question2'].astype(str).apply(clean)
df['is_duplicate'] = df['is_duplicate'].astype(float)

# ✅ Step 4: Stratified split into train/dev sets (90/10)
train_df, dev_df = train_test_split(
    df, test_size=0.1, stratify=df['is_duplicate'], random_state=42
)
train_samples = [InputExample(texts=[q1, q2], label=label)
                 for q1, q2, label in zip(train_df.question1, train_df.question2, train_df.is_duplicate)]
dev_samples = list(zip(dev_df.question1.tolist(), dev_df.question2.tolist(), dev_df.is_duplicate.tolist()))

# ✅ Step 5: Load SBERT model and set up training
model = SentenceTransformer('all-MiniLM-L6-v2')
train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=32)
train_loss = losses.CosineSimilarityLoss(model=model)

# ✅ Step 6: Training config
output_path = os.path.join("models", "sbert_quora_model")
os.makedirs(output_path, exist_ok=True)

num_epochs = 10
warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)
best_accuracy = 0
best_threshold = 0
epochs_no_improve = 0
patience = 3

# ✅ Step 7: Training loop with early stopping
for epoch in range(num_epochs):
    print(f"\n🚀 Epoch {epoch + 1}/{num_epochs} training...")

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=1,
        warmup_steps=warmup_steps,
        show_progress_bar=True,
        use_amp=True
    )

    # ✅ Evaluate on dev set
    pred_scores = []
    true_labels = []

    for q1, q2, label in dev_samples:
        emb1 = model.encode(q1, convert_to_tensor=True, show_progress_bar=False)
        emb2 = model.encode(q2, convert_to_tensor=True, show_progress_bar=False)
        score = float(util.cos_sim(emb1, emb2))
        pred_scores.append(score)
        true_labels.append(int(label))

    # ✅ Tune threshold
    best_epoch_threshold = 0
    best_epoch_acc = 0
    for threshold in np.arange(0.6, 0.9, 0.005):
        pred_labels = [1 if s > threshold else 0 for s in pred_scores]
        acc = accuracy_score(true_labels, pred_labels) * 100
        if acc > best_epoch_acc:
            best_epoch_acc = acc
            best_epoch_threshold = threshold

    print(f"✅ Accuracy after Epoch {epoch + 1}: {best_epoch_acc:.2f}% at threshold {best_epoch_threshold:.3f}")

    # ✅ Save best model
    if best_epoch_acc > best_accuracy:
        best_accuracy = best_epoch_acc
        best_threshold = best_epoch_threshold
        model.save(output_path)
        with open(os.path.join(output_path, 'best_threshold.txt'), 'w') as f:
            f.write(str(best_threshold))
        print("✅ Best model updated and saved.")
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        print(f"⚠️ No improvement. Early stop count: {epochs_no_improve}/{patience}")
        if epochs_no_improve >= patience:
            print("\n🛑 Early stopping triggered. Ending training.")
            break

# ✅ Final report
print(f"\n📊 Best Accuracy: {best_accuracy:.2f}%")
print(f"🎯 Best Threshold: {best_threshold:.3f}")
print(f"💾 Best Model Saved at: {output_path}")
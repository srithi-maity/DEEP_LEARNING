# =========================================================
# Simple Seq2Seq using LSTM & GRU for English to Hindi translation
# =========================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

# -----------------------------
# 1. Load Dataset
# -----------------------------
# Example: you can replace this path with your downloaded CSV
# Kaggle dataset file: 'Hindi_English_Truncated_Corpus.csv'
data = pd.read_csv("Hindi_English_Truncated_Corpus.csv")

# Keep only relevant columns
data = data[['english_sentence', 'hindi_sentence']].dropna().head(10000)

# Lowercase and add special tokens
data['english_sentence'] = data['english_sentence'].apply(lambda x: x.lower().strip())
data['hindi_sentence'] = data['hindi_sentence'].apply(lambda x: "<sos> " + x.strip() + " <eos>")

# Train-test split
train_df, val_df = train_test_split(data, test_size=0.1, random_state=42)

# -----------------------------
# 2. Tokenization and Vocabulary
# -----------------------------
from collections import Counter

def build_vocab(sentences):
    words = []
    for sent in sentences:
        words.extend(sent.split())
    vocab = {"<pad>": 0, "<unk>": 1}
    for i, word in enumerate(Counter(words).keys()):
        vocab[word] = i + 2
    return vocab

eng_vocab = build_vocab(train_df['english_sentence'])
hin_vocab = build_vocab(train_df['hindi_sentence'])

def encode_sentence(sentence, vocab):
    return [vocab.get(word, vocab["<unk>"]) for word in sentence.split()]

def collate_fn(batch):
    src_batch, trg_batch = [], []
    for src, trg in batch:
        src_batch.append(torch.tensor(encode_sentence(src, eng_vocab), dtype=torch.long))
        trg_batch.append(torch.tensor(encode_sentence(trg, hin_vocab), dtype=torch.long))
    src_batch = pad_sequence(src_batch, padding_value=0, batch_first=True)
    trg_batch = pad_sequence(trg_batch, padding_value=0, batch_first=True)
    return src_batch, trg_batch

# Custom Dataset
class TranslationDataset(Dataset):
    def __init__(self, df):
        self.df = df
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        return self.df.iloc[idx]['english_sentence'], self.df.iloc[idx]['hindi_sentence']

train_loader = DataLoader(TranslationDataset(train_df), batch_size=64, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(TranslationDataset(val_df), batch_size=64, shuffle=False, collate_fn=collate_fn)

# -----------------------------
# 3. Seq2Seq Model (Encoder–Decoder)
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EncoderLSTM(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hid_dim, n_layers, batch_first=True)
    def forward(self, src):
        embedded = self.embedding(src)
        outputs, (hidden, cell) = self.lstm(embedded)
        return hidden, cell

class DecoderLSTM(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hid_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hid_dim, output_dim)
    def forward(self, input, hidden, cell):
        input = input.unsqueeze(1)
        embedded = self.embedding(input)
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        prediction = self.fc(output.squeeze(1))
        return prediction, hidden, cell

class Seq2SeqLSTM(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.fc.out_features
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(device)

        hidden, cell = self.encoder(src)
        input = trg[:, 0]  # <sos>
        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[:, t, :] = output
            top1 = output.argmax(1)
            input = trg[:, t] if torch.rand(1).item() < teacher_forcing_ratio else top1
        return outputs

# -----------------------------
# 4. Training
# -----------------------------
INPUT_DIM = len(eng_vocab)
OUTPUT_DIM = len(hin_vocab)
EMB_DIM = 128
HID_DIM = 256

encoder = EncoderLSTM(INPUT_DIM, EMB_DIM, HID_DIM)
decoder = DecoderLSTM(OUTPUT_DIM, EMB_DIM, HID_DIM)
model = Seq2SeqLSTM(encoder, decoder).to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss(ignore_index=0)

EPOCHS = 5
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for src, trg in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        src, trg = src.to(device), trg.to(device)
        optimizer.zero_grad()
        output = model(src, trg)
        output_dim = output.shape[-1]
        output = output[:, 1:].reshape(-1, output_dim)
        trg = trg[:, 1:].reshape(-1)
        loss = criterion(output, trg)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1} Loss: {total_loss/len(train_loader):.4f}")

# -----------------------------
# 5. Inference Function
# -----------------------------
def translate_sentence(sentence):
    model.eval()
    tokens = sentence.lower().split()
    src_tensor = torch.tensor(encode_sentence(sentence.lower(), eng_vocab)).unsqueeze(0).to(device)
    hidden, cell = model.encoder(src_tensor)
    input = torch.tensor([hin_vocab["<sos>"]]).to(device)
    result = []
    for _ in range(20):
        output, hidden, cell = model.decoder(input, hidden, cell)
        top1 = output.argmax(1)
        if top1.item() == hin_vocab.get("<eos>"):
            break
        result.append(list(hin_vocab.keys())[list(hin_vocab.values()).index(top1.item())])
        input = top1
    return " ".join(result)

# -----------------------------
# 6. Test translation
# -----------------------------
print("\nExample translations:")
examples = [
    "how are you",
    "good morning",
    "i am a student",
]
for s in examples:
    print(f"{s} → {translate_sentence(s)}")

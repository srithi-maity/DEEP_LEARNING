import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
import warnings

warnings.filterwarnings('ignore')

# Download NLTK data (run once)
# nltk.download('punkt')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class TranslationDataset(Dataset):
    def __init__(self, english_sentences, hindi_sentences, max_length):
        self.english_sentences = english_sentences
        self.hindi_sentences = hindi_sentences
        self.max_length = max_length

    def __len__(self):
        return len(self.english_sentences)

    def __getitem__(self, idx):
        return self.english_sentences[idx], self.hindi_sentences[idx]


class Preprocessor:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.vocab_size = 0

    def build_vocab(self, sentences, min_freq=2):
        word_counts = Counter()
        for sentence in sentences:
            words = sentence.split()
            word_counts.update(words)

        # Create vocabulary
        self.word2idx = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3}
        idx = 4

        for word, count in word_counts.items():
            if count >= min_freq:
                self.word2idx[word] = idx
                idx += 1

        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        self.vocab_size = len(self.word2idx)

    def sentence_to_indices(self, sentence, max_length):
        words = sentence.split()
        indices = [self.word2idx.get(word, self.word2idx['<UNK>']) for word in words]

        # Add SOS and EOS tokens, and pad to max_length
        indices = [self.word2idx['<SOS>']] + indices[:max_length - 2] + [self.word2idx['<EOS>']]
        indices += [self.word2idx['<PAD>']] * (max_length - len(indices))

        return indices[:max_length]


class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, dropout, rnn_type='lstm'):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_type = rnn_type

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.dropout = nn.Dropout(dropout)

        if rnn_type == 'lstm':
            self.rnn = nn.LSTM(embed_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        else:  # gru
            self.rnn = nn.GRU(embed_size, hidden_size, num_layers, dropout=dropout, batch_first=True)

    def forward(self, x):
        # x shape: (batch_size, seq_length)
        embedded = self.dropout(self.embedding(x))
        # embedded shape: (batch_size, seq_length, embed_size)

        if self.rnn_type == 'lstm':
            outputs, (hidden, cell) = self.rnn(embedded)
            return outputs, hidden, cell
        else:
            outputs, hidden = self.rnn(embedded)
            return outputs, hidden, None


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, dropout, rnn_type='lstm'):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_type = rnn_type

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.dropout = nn.Dropout(dropout)

        if rnn_type == 'lstm':
            self.rnn = nn.LSTM(embed_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        else:
            self.rnn = nn.GRU(embed_size, hidden_size, num_layers, dropout=dropout, batch_first=True)

        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden, cell=None):
        # x shape: (batch_size, 1)
        x = x.unsqueeze(1)
        # x shape: (batch_size, 1)

        embedded = self.dropout(self.embedding(x))
        # embedded shape: (batch_size, 1, embed_size)

        if self.rnn_type == 'lstm':
            outputs, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        else:
            outputs, hidden = self.rnn(embedded, hidden)
            cell = None

        predictions = self.fc(outputs.squeeze(1))
        # predictions shape: (batch_size, vocab_size)

        return predictions, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, target, teacher_forcing_ratio=0.5):
        batch_size = source.shape[0]
        target_len = target.shape[1]
        target_vocab_size = self.decoder.fc.out_features

        # Initialize outputs
        outputs = torch.zeros(batch_size, target_len, target_vocab_size).to(device)

        # Encode source sequence
        encoder_outputs, hidden, cell = self.encoder(source)

        # First input to decoder is SOS token
        x = target[:, 0]

        for t in range(1, target_len):
            # Decode
            output, hidden, cell = self.decoder(x, hidden, cell)
            outputs[:, t] = output

            # Teacher forcing
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(1)
            x = target[:, t] if teacher_force else top1

        return outputs


def load_and_preprocess_data():
    # For demonstration, creating synthetic data
    # In practice, load from the Kaggle dataset
    english_sentences = [
        "I am going to school",
        "She is reading a book",
        "They are playing football",
        "The sun is shining",
        "We are learning machine translation",
        "He works in an office",
        "The cat is sleeping",
        "I love programming",
        "She sings beautifully",
        "They are cooking dinner"
    ]

    hindi_sentences = [
        "मैं स्कूल जा रहा हूँ",
        "वह एक किताब पढ़ रही है",
        "वे फुटबॉल खेल रहे हैं",
        "सूरज चमक रहा है",
        "हम मशीन अनुवाद सीख रहे हैं",
        "वह एक ऑफिस में काम करता है",
        "बिल्ली सो रही है",
        "मुझे प्रोग्रामिंग पसंद है",
        "वह खूबसूरती से गाती है",
        "वे रात का खाना बना रहे हैं"
    ]

    return english_sentences, hindi_sentences


def train_model(model, dataloader, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0

    for i, (src, trg) in enumerate(dataloader):
        src = src.to(device)
        trg = trg.to(device)

        optimizer.zero_grad()
        output = model(src, trg)

        # Reshape for loss calculation
        output_dim = output.shape[-1]
        output = output[:, 1:].reshape(-1, output_dim)
        trg = trg[:, 1:].reshape(-1)

        loss = criterion(output, trg)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(dataloader)


def evaluate_model(model, dataloader, criterion):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for i, (src, trg) in enumerate(dataloader):
            src = src.to(device)
            trg = trg.to(device)

            output = model(src, trg, 0)  # No teacher forcing

            output_dim = output.shape[-1]
            output = output[:, 1:].reshape(-1, output_dim)
            trg = trg[:, 1:].reshape(-1)

            loss = criterion(output, trg)
            epoch_loss += loss.item()

    return epoch_loss / len(dataloader)


def demonstrate_vanishing_gradient():
    """Demonstrate how LSTM and GRU solve vanishing gradient problem"""

    # Simple RNN vs LSTM/GRU gradient flow demonstration
    seq_length = 50
    input_size = 10
    hidden_size = 20

    # Simple RNN
    rnn = nn.RNN(input_size, hidden_size, batch_first=True)

    # LSTM
    lstm = nn.LSTM(input_size, hidden_size, batch_first=True)

    # GRU
    gru = nn.GRU(input_size, hidden_size, batch_first=True)

    # Test input
    x = torch.randn(1, seq_length, input_size)

    # Compute gradients for each type
    models = {'RNN': rnn, 'LSTM': lstm, 'GRU': gru}
    gradient_norms = {}

    for name, model in models.items():
        # Forward pass
        if name == 'LSTM':
            output, (hidden, cell) = model(x)
            # Backward pass
            hidden.mean().backward()
        else:
            output, hidden = model(x)
            # Backward pass
            hidden.mean().backward()

        # Calculate gradient norms
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        gradient_norms[name] = total_norm

        # Zero gradients
        model.zero_grad()

    return gradient_norms


# Main execution
def main():
    # Load data
    english_sentences, hindi_sentences = load_and_preprocess_data()

    # Preprocessors
    eng_preprocessor = Preprocessor()
    hin_preprocessor = Preprocessor()

    # Build vocabularies
    eng_preprocessor.build_vocab(english_sentences)
    hin_preprocessor.build_vocab(hindi_sentences)

    print(f"English vocabulary size: {eng_preprocessor.vocab_size}")
    print(f"Hindi vocabulary size: {hin_preprocessor.vocab_size}")

    # Convert sentences to indices
    max_length = 20
    eng_indices = [eng_preprocessor.sentence_to_indices(sent, max_length) for sent in english_sentences]
    hin_indices = [hin_preprocessor.sentence_to_indices(sent, max_length) for sent in hindi_sentences]

    # Create dataset and dataloader
    dataset = TranslationDataset(
        torch.tensor(eng_indices, dtype=torch.long),
        torch.tensor(hin_indices, dtype=torch.long),
        max_length
    )

    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # Hyperparameters
    EMBED_SIZE = 256
    HIDDEN_SIZE = 512
    NUM_LAYERS = 2
    DROPOUT = 0.5
    LEARNING_RATE = 0.001
    CLIP = 1
    NUM_EPOCHS = 50

    # Train LSTM model
    print("\nTraining LSTM model...")
    encoder_lstm = Encoder(eng_preprocessor.vocab_size, EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS, DROPOUT, 'lstm').to(device)
    decoder_lstm = Decoder(hin_preprocessor.vocab_size, EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS, DROPOUT, 'lstm').to(device)
    model_lstm = Seq2Seq(encoder_lstm, decoder_lstm).to(device)

    optimizer_lstm = optim.Adam(model_lstm.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding index

    lstm_train_losses = []
    lstm_val_losses = []

    for epoch in range(NUM_EPOCHS):
        train_loss = train_model(model_lstm, dataloader, optimizer_lstm, criterion, CLIP)
        val_loss = evaluate_model(model_lstm, dataloader, criterion)

        lstm_train_losses.append(train_loss)
        lstm_val_losses.append(val_loss)

        if epoch % 10 == 0:
            print(f'LSTM Epoch: {epoch + 1:02} | Train Loss: {train_loss:.3f} | Val Loss: {val_loss:.3f}')

    # Train GRU model
    print("\nTraining GRU model...")
    encoder_gru = Encoder(eng_preprocessor.vocab_size, EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS, DROPOUT, 'gru').to(device)
    decoder_gru = Decoder(hin_preprocessor.vocab_size, EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS, DROPOUT, 'gru').to(device)
    model_gru = Seq2Seq(encoder_gru, decoder_gru).to(device)

    optimizer_gru = optim.Adam(model_gru.parameters(), lr=LEARNING_RATE)

    gru_train_losses = []
    gru_val_losses = []

    for epoch in range(NUM_EPOCHS):
        train_loss = train_model(model_gru, dataloader, optimizer_gru, criterion, CLIP)
        val_loss = evaluate_model(model_gru, dataloader, criterion)

        gru_train_losses.append(train_loss)
        gru_val_losses.append(val_loss)

        if epoch % 10 == 0:
            print(f'GRU Epoch: {epoch + 1:02} | Train Loss: {train_loss:.3f} | Val Loss: {val_loss:.3f}')

    # Plot results
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.plot(lstm_train_losses, label='LSTM Train')
    plt.plot(lstm_val_losses, label='LSTM Val')
    plt.plot(gru_train_losses, label='GRU Train')
    plt.plot(gru_val_losses, label='GRU Val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)

    # Demonstrate vanishing gradient solution
    gradient_norms = demonstrate_vanishing_gradient()

    plt.subplot(1, 2, 2)
    plt.bar(gradient_norms.keys(), gradient_norms.values())
    plt.xlabel('RNN Type')
    plt.ylabel('Gradient Norm')
    plt.title('Gradient Flow Comparison')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Print comparison results
    print("\n" + "=" * 50)
    print("PERFORMANCE COMPARISON")
    print("=" * 50)
    print(f"LSTM Final Train Loss: {lstm_train_losses[-1]:.4f}")
    print(f"LSTM Final Val Loss: {lstm_val_losses[-1]:.4f}")
    print(f"GRU Final Train Loss: {gru_train_losses[-1]:.4f}")
    print(f"GRU Final Val Loss: {gru_val_losses[-1]:.4f}")

    print("\nGRADIENT FLOW ANALYSIS:")
    for model_type, norm in gradient_norms.items():
        print(f"{model_type}: Gradient Norm = {norm:.6f}")

    print("\nEXPLANATION:")
    print("1. LSTM uses input, output, and forget gates with cell state to maintain long-term memory")
    print("2. GRU uses update and reset gates for a simpler architecture")
    print("3. Both maintain better gradient flow compared to simple RNN")
    print("4. Higher gradient norms indicate better prevention of vanishing gradients")


if __name__ == "__main__":
    main()
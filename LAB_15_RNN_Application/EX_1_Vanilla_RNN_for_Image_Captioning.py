import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import torchvision.transforms as transforms

class ImprovedCNN(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(128, embed_size)

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class ImprovedRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size + embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        embeddings = self.embed(captions)
        features = features.unsqueeze(1).repeat(1, embeddings.size(1), 1)
        rnn_input = torch.cat([embeddings, features], dim=2)
        outputs, _ = self.lstm(rnn_input)
        return self.fc(outputs)


class ImprovedCaptioner(nn.Module):
    def __init__(self, vocab_size, embed_size=128, hidden_size=256):
        super().__init__()
        self.encoder = ImprovedCNN(embed_size)
        self.decoder = ImprovedRNN(vocab_size, embed_size, hidden_size)

    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs

class CIFAR10CaptionDataset(Dataset):
    def __init__(self, train=True, max_samples=5000):
        self.train = train
        self.max_samples = max_samples

        # CIFAR-10 classes as "captions"
        self.classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                        'dog', 'frog', 'horse', 'ship', 'truck']

        self.vocab = {
            '<pad>': 0, '<start>': 1, '<end>': 2,
            'this': 3, 'is': 4, 'a': 5, 'photo': 6, 'of': 7
        }
        for i, cls in enumerate(self.classes):
            self.vocab[cls] = i + 8

        self.idx_to_word = {v: k for k, v in self.vocab.items()}

        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ])

        from torchvision.datasets import CIFAR10
        self.dataset = CIFAR10(root='./data', train=train, download=True, transform=self.transform)

        if max_samples < len(self.dataset):
            indices = torch.randperm(len(self.dataset))[:max_samples]
            self.dataset = torch.utils.data.Subset(self.dataset, indices)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        class_name = self.classes[label]
        caption = [1, 3, 4, 5, 6, 7, self.vocab[class_name], 2]
        caption = caption + [0] * (15 - len(caption))

        return image, torch.tensor(caption), label


def calculate_accuracy(predictions, targets, ignore_index=0):
    mask = targets != ignore_index
    correct = (predictions == targets) & mask
    accuracy = correct.sum().float() / mask.sum().float()
    return accuracy.item()


def train_real_dataset():
    train_dataset = CIFAR10CaptionDataset(train=True, max_samples=5000)
    val_dataset = CIFAR10CaptionDataset(train=False, max_samples=1000)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    model = ImprovedCaptioner(len(train_dataset.vocab), embed_size=128, hidden_size=256)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print(f"Training on {len(train_dataset)} real images")
    print(f"Vocabulary size: {len(train_dataset.vocab)}")
    print(f"Classes: {train_dataset.classes}")

    best_accuracy = 0

    for epoch in range(50):
        # Training
        model.train()
        train_loss = 0
        train_acc = 0

        for images, captions, labels in train_loader:
            outputs = model(images, captions[:, :-1])
            loss = criterion(outputs.reshape(-1, len(train_dataset.vocab)),
                             captions[:, 1:].reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # accuracy
            with torch.no_grad():
                predictions = outputs.argmax(-1)
                acc = calculate_accuracy(predictions, captions[:, 1:])
                train_acc += acc

        # Validation
        model.eval()
        val_loss = 0
        val_acc = 0
        val_samples = 0

        with torch.no_grad():
            for images, captions, labels in val_loader:
                outputs = model(images, captions[:, :-1])
                loss = criterion(outputs.reshape(-1, len(train_dataset.vocab)),
                                 captions[:, 1:].reshape(-1))
                val_loss += loss.item()

                predictions = outputs.argmax(-1)
                acc = calculate_accuracy(predictions, captions[:, 1:])
                val_acc += acc
                val_samples += 1

        train_loss /= len(train_loader)
        train_acc /= len(train_loader)
        val_loss /= len(val_loader)
        val_acc /= val_samples

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}:")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc * 100:.2f}%")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc * 100:.2f}%")

            if val_acc > best_accuracy:
                best_accuracy = val_acc
                torch.save(model.state_dict(), 'best_caption_model.pth')
                print(f"  ✓ New best model saved!")

    return model, train_dataset.vocab, train_dataset.idx_to_word, train_dataset.classes


def test_real_model(model, idx_to_word, classes):
    test_dataset = CIFAR10CaptionDataset(train=False, max_samples=10)

    model.eval()

    for i in range(min(5, len(test_dataset))):
        image, caption, true_label = test_dataset[i]
        true_class = classes[true_label]

        with torch.no_grad():
            features = model.encoder(image.unsqueeze(0))
            generated_caption = [1]  # start token 

            for _ in range(10):
                input_seq = torch.tensor([generated_caption])
                outputs = model.decoder(features, input_seq)
                next_word = outputs.argmax(2)[:, -1].item()
                if next_word == 2: break  # end token
                generated_caption.append(next_word)

            # Converting to words
            words = [idx_to_word[idx] for idx in generated_caption[1:]]
            generated_text = ' '.join(words)

        # To Visualize the plots
        plt.figure(figsize=(10, 4))

        plt.subplot(1, 2, 1)
        plt.imshow(image.permute(1, 2, 0).numpy())
        plt.title(f"True: {true_class}")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.text(0.1, 0.5, f"Generated:\n{generated_text}",
                 fontsize=12, va='center', wrap=True)
        plt.axis('off')

        plt.tight_layout()
        plt.show()

        print(f"Sample {i + 1}: True='{true_class}', Generated='{generated_text}'")


if __name__ == "__main__":
    print("Training on CIFAR-10 dataset...")
    model, vocab, idx_to_word, classes = train_real_dataset()

    print("\n" + "=" * 60)
    print("Testing on real CIFAR-10 images")
    print("=" * 60)

    test_real_model(model, idx_to_word, classes)

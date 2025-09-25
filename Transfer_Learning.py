
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
from torchvision import datasets, models, transforms
import pandas as pd
from sklearn.svm import SVC

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# Normalization
# -------------------------------


transform_train = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
])

transform_test = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# -------------------------------
# Data Loading
# -------------------------------

training_data = datasets.CIFAR10(
    root="data",
    train=True,
    download=False,
    transform=transform_train,
)
trainloader = torch.utils.data.DataLoader(training_data, batch_size=64, shuffle=True)


test_data = datasets.CIFAR10(
    root="data",
    train=False,
    download=False,
    transform=transform_test,
)
testloader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=False)


# -------------------------------
# Pretrained Model
# -------------------------------


def Feature_extractor_and_Linear_Layer ():
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad=False
    model.fc=nn.Linear(model.fc.in_features,10)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)


    # -------------------------------
    # Training loop
    # -------------------------------
    epochs=5
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(trainloader):.4f}")

    # -------------------------------
    # Evaluation
    # -------------------------------
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Test Accuracy with Feature Extractor and Linear Layer : {100 * correct / total:.2f}%")

def Fine_tuning_and_Linear_Layer():
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad=True
    model.fc=nn.Linear(model.fc.in_features,10)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

    # -------------------------------
    # Training loop
    # -------------------------------
    epochs = 5
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(trainloader):.4f}")

    # -------------------------------
    # Evaluation
    # -------------------------------
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Test Accuracy with Fine Tuning and Linear Layer : {100 * correct / total:.2f}%")

    # Epoch[1 / 5], Loss: 0.8287
    # Epoch[2 / 5], Loss: 0.6204
    # Epoch[3 / 5], Loss: 0.5883
    # Epoch[4 / 5], Loss: 0.5770
    # Epoch[5 / 5], Loss: 0.5723
    # Test
    # Accuracy
    # with Fine Tuning and Linear Layer: 80.41 %




def extract_features(dataloader,feature_extractor):
    feats, labels = [], []
    with torch.no_grad():
        for images, lbls in dataloader:
            images = images.to(device)
            outputs = feature_extractor(images)   # (batch, 512)
            outputs = outputs.view(outputs.size(0), -1)
            feats.append(outputs.cpu().numpy())
            labels.append(lbls.numpy())
    feats = np.concatenate(feats, axis=0)
    labels = np.concatenate(labels, axis=0)
    return feats, labels


def Feature_extractor_and_SVM():

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    # last_layer = list(model.children())[-1]
    # print(last_layer)
    print(model.fc)

    feature_extractor = torch.nn.Sequential(*list(model.children())[:-1]).to(device)

    x_train,y_train = extract_features(trainloader,feature_extractor)
    X_test, y_test = extract_features(testloader,feature_extractor)


    # df = pd.DataFrame(x_train)
    # df["label"] = y_train
    # df.to_csv("cifar10_features.csv", index=False)
    # print("Feature shape:", x_train.shape)     # Feature shape: (50000, 512)

    #--------------------------------------
    #Train SVM
    #--------------------------------------

    print("Training SVM classifier...")
    SVM = SVC(kernel='rbf', C=10, gamma='scale')
    SVM.fit(x_train, y_train)


    # -------------------------------
    # Evaluation
    # -------------------------------

    acc = SVM.score(X_test, y_test)
    print(f"SVM Test Accuracy: {acc * 100:.2f}%")


#  ________________________RESULT______________________________

    #Linear(in_features=512, out_features=1000, bias=True)
    # Training SVM classifier...
    # SVM Test Accuracy: 89.33 %


def Fine_Tuning_and_Freeze_few_Layer():
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model = model.to(device)


    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad= True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

    # -------------------------------
    # Training loop
    # -------------------------------
    epochs = 5
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(trainloader):.4f}")

    # -------------------------------
    # Evaluation
    # -------------------------------
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Test Accuracy with Fine Tuning and Freeze few Layers : {100 * correct / total:.2f}%")


if __name__=="__main__":
    # Fine_tuning_and_Linear_Layer()
    # Feature_extractor_and_SVM()
    Fine_Tuning_and_Freeze_few_Layer()

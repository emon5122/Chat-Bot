import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from model import NeuralNet
from nltk_utils import bag_of_words, stem, tokenize

IGNORED_CHARS = ["?", "!", ".", ","]
with open("data.json", "r") as f:
    data = json.load(f)

all_words = []
tags = []
xy = []
intents=data.get("intents", [])
commands = data.get("commands", [])
personality_traits = data.get("personality_traits", [])
for intent in intents:
    tag = intent["tag"]
    tags.append(tag)
    for pattern in intent["patterns"]:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

for command in commands:
    tag = command["action"]
    tags.append(tag)
    for pattern in command["patterns"]:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

for trait in personality_traits:
    tag = trait["trait"]
    tags.append(tag)
    for response in trait["responses"]:
        w = tokenize(response)
        all_words.extend(w)
        xy.append((w, tag))

all_words = [stem(w) for w in all_words if w not in IGNORED_CHARS]
all_words = sorted(set(all_words))
tags = sorted(set(tags))
X_train = []
Y_train = []
for pattern_sentence, tag in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    label = tags.index(tag)
    Y_train.append(label)

X_train = np.array(X_train)
Y_train = np.array(Y_train)


class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = Y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples


dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=8, shuffle=True, num_workers=2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# hyperparameters
num_epochs = 1000
input_size = len(X_train[0])
hidden_size = 8
output_size = len(tags)
learning_rate = 0.001
model = NeuralNet(
    input_size=input_size, hidden_size=hidden_size, num_classes=output_size
).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


for epoch in range(num_epochs):
    for words, labels in train_loader:
        words = words.to(device)
        labels = labels.to(device)
        outputs = model(words)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if (epoch + 1) % 100 == 0:
        print(f"epoch {epoch+1}/{num_epochs}, loss={loss.item():.4f}")
print(f"final loss, loss={loss.item():.4f}")
data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "all_words": all_words,
    "tags": tags,
}
torch.save(data, "data.pth")
print("training complete. file saved to data.pth")

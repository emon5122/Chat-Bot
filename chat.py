import json
import random

import torch

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
with open("data.json", "r") as f:
    data = json.load(f)
MODEL = "data.pth"
model_data = torch.load(MODEL)
input_size = model_data["input_size"]
hidden_size = model_data["hidden_size"]
output_size = model_data["output_size"]
all_words = model_data["all_words"]
tags = model_data["tags"]
model_state = model_data["model_state"]
model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Emon"
print("Let's chat! Type 'quit' to exit")
while True:
    sentence = input("You: ")
    if sentence == "quit":
        break
    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        # Check if the tag corresponds to an intent
        for intent in data["intents"]:
            if tag == intent["tag"]:
                print(f"{bot_name}: {random.choice(intent['responses'])}")
                break  # Break out of the loop once a matching intent is found
        else:
            # If the tag doesn't match any intent, check for commands
            for command in data.get("commands", []):
                if tag == command["action"]:
                    print(f"{bot_name}: {random.choice(command['responses'])}")
                    break  # Break out of the loop once a matching command is found
            else:
                # If the tag doesn't match any command, check for personality traits
                for trait in data.get("personality_traits", []):
                    if tag == trait["trait"]:
                        print(f"{bot_name}: {random.choice(trait['responses'])}")
                        break  # Break out of the loop once a matching personality trait is found
                else:
                    print(f"{bot_name}: I do not understand...")
    else:
        print(f"{bot_name}: I do not understand...")

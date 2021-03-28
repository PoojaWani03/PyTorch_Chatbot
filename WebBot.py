import random
import json
from flask import Flask,render_template,request
import torch

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

app = Flask(__name__)



bot_name = "Sam"
print("Let's chat! (type 'quit' to exit)")

@app.route('/',methods=['POST','GET'])
def takeMsg():
    if request.method == "POST":
        msg = request.form
        message=None
        for _ in msg.values():
            message=_
            break
        print(message)
        sentence=message
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
            for intent in intents['intents']:
                if tag == intent["tag"]:
                    print(f"{bot_name}: {random.choice(intent['responses'])}")
                    output = random.choice(intent['responses'])
        else:
            print(f"{bot_name}: I do not understand...")
            output="I do not understand..."
        return render_template("chatBot.html",answer=str(output),tag=tag)
    else:
        return render_template("chatBot.html",answer="Hi I am sam, Chat with me!")

    return render_template


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open('intents.json', 'r') as json_data:
        intents = json.load(json_data)

    FILE = "data.pth"
    data = torch.load(FILE,map_location=torch.device('cpu'))

    input_size = data["input_size"]
    hidden_size = data["hidden_size"]
    output_size = data["output_size"]
    all_words = data['all_words']
    tags = data['tags']
    model_state = data["model_state"]

    model = NeuralNet(input_size, hidden_size, output_size).to(device)
    model.load_state_dict(model_state)
    model.eval()
    app.run(debug=True)
# while True:
#     # sentence = "do you use credit cards?"
#     sentence = input("You: ")
#     if sentence == "quit":
#         break
#
#     sentence = tokenize(sentence)
#     X = bag_of_words(sentence, all_words)
#     X = X.reshape(1, X.shape[0])
#     X = torch.from_numpy(X).to(device)
#
#     output = model(X)
#     _, predicted = torch.max(output, dim=1)
#
#     tag = tags[predicted.item()]
#
#     probs = torch.softmax(output, dim=1)
#     prob = probs[0][predicted.item()]
#     if prob.item() > 0.75:
#         for intent in intents['intents']:
#             if tag == intent["tag"]:
#                 print(f"{bot_name}: {random.choice(intent['responses'])}")
#     else:
#         print(f"{bot_name}: I do not understand...")

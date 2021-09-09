from __future__ import unicode_literals, print_function, division
from io import open
import glob
import unicodedata
import string
import torch
import torch.nn as nn
from torch.autograd import Variable
from flask import Flask, request
app = Flask(__name__)

all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)
category_lines = {}
all_categories = []
n_hidden = 128

findFiles = lambda path: glob.glob(path)
unicodeToAscii = lambda s: ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn' and c in all_letters)
readLines = lambda filename: [unicodeToAscii(line) for line in open(filename, encoding='utf-8').read().strip().split()]
letterToIndex = lambda letter: all_letters.find(letter)
for filename in findFiles('data/names/*.txt'):
    category = filename.split('/')[-1].split('.')[0]
    all_categories.append(category)
    lines = readLines(filename)
    category_lines[category] = lines

n_categories = len(all_categories)

def letterToTensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letterToIndex(letter)] = 1
    return tensor

def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        return output, hidden

    def initHidden(self):
        return Variable(torch.zeros(1, self.hidden_size))

rnn = RNN(n_letters, n_hidden, n_categories)

def evaluate(line_tensor):
    hidden = rnn.initHidden()
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)
    return output

def predict(input_line, n_predictions=3):
    with torch.no_grad():
        output = evaluate(lineToTensor(input_line))
        topv, topi = output.topk(n_predictions, 1, True)
        predictions = [[topv[0][i].item(), all_categories[topi[0][i].item()]] for i in range(n_predictions)]
    return predictions

@app.route('/invoke', methods=['POST'])
def invoke():
    return {'result': predict(request.get_data().decode("utf-8"))}


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9000)

import flask
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from flask import request
app = flask.Flask(__name__)

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

@app.route('/', methods=['GET'])
def home():
    return 'api working'

@app.route('/talk/', methods=['POST'])
def talk():
    if request.form.__getitem__('text'):
        # initialize tokenizer and model from pretrained GPT2 model
        inputs = tokenizer.encode(request.form.__getitem__('text'), return_tensors='pt')
        outputs = model.generate(inputs, max_length=200, do_sample=False)
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return text
    else: 
        return '500'

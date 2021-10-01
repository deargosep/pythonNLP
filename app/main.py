import flask
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from flask import request
app = flask.Flask(__name__)
app.config["DEBUG"] = True


@app.route('/', methods=['GET'])
def home():
    return "<h1>Distant Reading Archive</h1><p>This site is a prototype API for distant reading of science fiction novels.</p>"

@app.route('/talk/', methods=['POST'])
def talk():
    if request.form.__getitem__('text'):
        # initialize tokenizer and model from pretrained GPT2 model
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        model = GPT2LMHeadModel.from_pretrained('gpt2')
        inputs = tokenizer.encode(request.form.__getitem__('text'), return_tensors='pt')
        outputs = model.generate(inputs, max_length=200, do_sample=True)
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(text)
        return text
    else: 
        print('no request data?')
        return '500'

app.run()
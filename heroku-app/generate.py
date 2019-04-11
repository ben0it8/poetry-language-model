from random import sample
from flask import Flask, Response, json, render_template, jsonify
from flask_cors import CORS
import torch
from utils import generate_line, parse_last_line, is_unbalanced, multiline, parse_models

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
CORS(app)

def generate_text(model, tokenizer, num_lines=10, min_len=8, max_len=15,
                  unk_id=0, sos_id=1, eos_id=2, temp=0.55):
  model.eval()  
  lines = []
  line_cnt = 0
  hidden = model.init_hidden(1)
  with torch.no_grad():
    
    while line_cnt != num_lines:
      try:
        ids, hidden = generate_line(model, hidden=hidden, temp=temp, max_len=max_len,
                                    sos_id=sos_id, eos_id=eos_id, unk_id=unk_id)
        
        if len(ids) <= min_len: raise Exception
        line = tokenizer(ids).strip()
        
        if line.startswith(tuple("-?!.,:()")): raise Exception
        if is_unbalanced(line): raise Exception
        
        lines += [line.capitalize()]
        line_cnt +=1  
      except: pass

  lines = parse_last_line(lines)
  return "\n".join(lines)

@app.route('/')
def main(name='Poetry Day 2018'):
  return render_template('generator.html', name=name)

@app.route('/generate/<name>', methods=['GET'])
@app.route('/generate/<name>/<float:temp>', methods=['GET'])
@multiline
def generate(name, temp=0.55):
  if temp > 1.0: temp=1.0
  if temp < 0.3: temp=0.3

  model, tokenizer, params = parse_models(name)
  text = generate_text(model, tokenizer, temp=temp, **params)
  del model, tokenizer
  import gc; gc.collect() # free up memory after inference is done
  return text

if __name__ == "__main__":
    app.run(debug=False)

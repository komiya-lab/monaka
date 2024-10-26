import os
from flask import Flask, request
from monaka.predictor import Predictor, LemmaPredictor, RESC_DIR, Encoder, Decoder

app = Flask(__name__)
MODELS = dict()

@app.route("/")
def index():
    return "hello"

@app.route("/model/<modelname>/dic/<dicname>/parse", methods=['POST'])
def parse2json(modelname, dicname):
    if modelname in MODELS:
        model = MODELS[modelname]
    else:
        model = Predictor(model_dir=os.path.join(RESC_DIR, modelname))

    sentence = request.json.get("sentence", [''])
    if isinstance(sentence, str):
        sentence = [sentence]

    return model.predict(sentence, suw_tokenizer='mecab', suw_tokenizer_option={"dic": dicname}, device='cpu', batch_size=1, encoder_name=request.json.get("output_format", 'jsonl'),
        node_format=request.json.get("node_format", '%m\t%f[9]\t%f[6]\t%f[7]\t%F-[0,1,2,3]\t%f[4]\t%f[5]\t%f[13]\t%f[27]\t%f[28]\n'), 
        unk_format=request.json.get("unk_format", '%m\t%m\t%m\t%m\tUNK\t%f[4]\t%f[5]\t\n'), 
        eos_format=request.json.get("eos_format", 'EOS\n'), 
        bos_format=request.json.get("bos_format", '')
    )
    

if __name__ == '__main__':
    app.run()
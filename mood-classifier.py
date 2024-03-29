from flask import Flask, request,jsonify
from transformers import pipeline
import os

app=Flask(__name__)


@app.route('/')
def reset():
    return "Use /hme"

@app.route('/hme',methods=['GET','POST'])
def classifier():

    print(r"Inside /hme")
    if not os.path.exists('output_dir/tf_model.h5'):
        print("Downloading model")
        os.system('sh download_model.sh')
        print("Downloaded model")
    print("Setting mood object")
    mood=pipeline('sentiment-analysis',model='output_dir')

    try:
        print("Taking input")
        inpText=request.json
        print(inpText)
        return jsonify(mood(inpText['Text'])[0].get('label'))
    except:
         print("Error while requesting input")

if __name__=='__main__':
    # from waitress import serve
    # serve(app, host="0.0.0.0", port=8000)
    app.run()
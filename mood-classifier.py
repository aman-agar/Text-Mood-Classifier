from flask import Flask, request,jsonify
from transformers import pipeline
import os

app=Flask(__name__)

mood=pipeline('sentiment-analysis',model='./output_dir')
if not os.path.exists('output_dir/tf_model.h5'):
        os.system('sh download_model.sh')

@app.route('/')
def reset():
    return "Use /hme"

@app.route('/hme',methods=['GET','POST'])
def classifier():
    try:
        inpText=request.json
        print(inpText)
        return jsonify(mood(inpText['Text'])[0].get('label'))
    except:
         print("Error while requesting input")

if __name__=='__main__':
    from waitress import serve
    serve(app, host="0.0.0.0", port=8090)
    
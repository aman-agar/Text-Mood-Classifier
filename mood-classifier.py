from flask import Flask, request,jsonify
from transformers import pipeline
import os

app=Flask(__name__)

mood=pipeline('sentiment-analysis',model='./output_dir')
if not os.path.exists('output_dir/tf_model.h5'):
        os.system('sh download_model.sh')

@app.route('/',methods=['GET','POST'])
def classifier():

    if request.method=='POST':
        inpText=request.json
        print(inpText)
        return jsonify(mood(inpText['Text'])[0].get('label'))

if __name__=='__main__':
    app.run()
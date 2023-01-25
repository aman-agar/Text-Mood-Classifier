from flask import Flask, request,jsonify
from transformers import pipeline

app=Flask(__name__)

mood=pipeline('sentiment-analysis',model='./output_dir')
@app.route('/',methods=['GET','POST'])
def classifier():

    if request.method=='POST':
        inpText=request.json
        print(inpText)
        return jsonify(mood(inpText['Text'])[0].get('label'))

if __name__=='__main__':
    app.run(debug=True,port=5000)
from transformers import RobertaTokenizerFast, TFRobertaForSequenceClassification, pipeline
from flask import Flask,request
# from tensorflow import load_model

#Download pre-trained model on the first run
# tokenizer = RobertaTokenizerFast.from_pretrained("arpanghoshal/EmoRoBERTa")
# model = TFRobertaForSequenceClassification.from_pretrained("arpanghoshal/EmoRoBERTa")

# model=load_model('tf_model.h5')
app=Flask(__name__)

# Create pipeline for running the model
print("Loaded Model!")
mood = pipeline('sentiment-analysis', model='arpanghoshal/EmoRoBERTa')
# print(mood("I dont want it"))


@app.route('/test')
def trial():
    return "Congrats! Trial running successfully!!"


@app.route('/',methods=['GET','POST'])
def classifier():

    inpText=request.json
    print(inpText)

    emotion_labels = (mood(inpText['Text'])[0].get('label'))
    print("Got emotion!")
    return emotion_labels

if __name__=='__main__':
    app.run()
from flask import Flask ,render_template ,request
from io import BytesIO
import base64, json
import numpy as np
from PIL import Image
from datetime import datetime
from PIL import  Image
from keras.models import load_model

model = load_model('model/collabmnist.h5')
app= Flask(__name__)

def value_invert(img):
    
    flatimg = img.flatten()
    for i in range(flatimg.size):
        flatimg[i] = 1 - flatimg[i]

    return flatimg.reshape(img.shape)

def predict_digit(img):
    img=Image.open(img)
    #resize image to 28x28 pixels
    img = img.resize((28,28))
    #convert rgb to grayscale
    img = img.convert('L')
    img = np.array(img)
    #reshaping to support our model input and normalizing also invert to black canvas with white paint as mnist has data in that format
    img=value_invert(img/255).reshape((1, 28, 28,1))
    
    res = model.predict([img])[0]
    return np.argmax(res), max(res)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/DigitRecognition',methods=['GET','POST'])
def Execpy():
    result = {"predict_digit" :"!", "detect_img" :"", "accuracy" :"", "prob" :{}}
    if request.method == 'POST' :
        # retrive the image
        postImg = BytesIO(base64.urlsafe_b64decode(request.form['img']))
        
        
        #predict the digit
        digit, accuracy = predict_digit(postImg)
        result['predict_digit']= str(digit)
        result['accuracy']=str(accuracy)
        
        
        '''
        #save predicted image 
    
        postImg = Image.open(postImg)
        postImg.save("./saveImage/{}_save.png".format(datetime.now().strftime('%m-%d_%H.%M.%S')))
        '''
        
    return json.dumps(result)
    

if __name__== '__main__':
    app.run("0.0.0.0",5000,threaded=False)
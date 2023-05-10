import json
from flask import Flask, request, json
import numpy as np
import base64
import cv2
import unidecode
import traceback

app = Flask(__name__)

# TODO
# Import model
from predict import TextSystem
from utils import parse_args

args = parse_args()
text_sys = TextSystem(args)

with open("models/dict_translate.json", "r", encoding="utf-8") as f:
    dict_translate = json.load(f)

app = Flask(__name__)

#warmup 
for _ in range(3):
    img = cv2.imread("images/001.png")
    text_sys.mapping(img, "warmup")

# Health-checking method
@app.route('/healthCheck', methods=['GET'])
def health_check():
    """
    Health check the server
    Return:
    Status of the server
        "OK"
    """
    return "OK"

# Inference method
@app.route('/infer', methods=['POST'])
def infer():
    """
    Do inference on input image
    Return:
    Dictionary Object following this schema
        {
            "image_name": <Image Name>
            "infers":
            [
                {
                    "food_name_en": <Food Name in Englist>
                    "food_name_vi": <Food Name in Vietnamese>
                    "food_price": <Price of food>
                }
            ]
        }
    """

    # Read data from request
    image_name = request.form.get('image_name')
    encoded_img = request.form.get('image')

    print("REQUEST:", image_name)

    # Convert base64 back to bytes
    img = base64.b64decode(encoded_img)
    img = np.frombuffer(img, dtype=np.uint8)
    img = cv2.imdecode(img, flags=1)
    # TODO
    # Call model for inference
    response = {
        "image_name": image_name,
        "infers": []
    }
    try:
        pairs = text_sys.mapping(img, image_name)
        for _, pair in pairs.iterrows():
            vi_name = spellCorrect(pair["VietnameseName"]).upper()
            if vi_name in dict_translate:
                en_name = dict_translate[vi_name]
            else:
                en_name = ""
            dct = {
                'food_name_en': en_name,
                'food_name_vi': vi_name,
                'food_price': pair["Price"].upper().replace('K', '000')
            }
            response['infers'].append(dct)
        return json.dumps(response)
        
    except Exception as e:
        print(traceback.format_exc())
        print(e)
        return json.dumps(response)
    

if __name__ == "__main__":
    app.run(debug=True, port=5000, host='0.0.0.0', use_reloader=False)
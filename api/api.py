import os
import io
import time
from typing import Union
from fastapi import FastAPI, File, UploadFile
from starlette.responses import RedirectResponse
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import cv2
import base64
from preprocess.scan import scan_img
from typing import List
import uvicorn
from pydantic import BaseModel

from predict import TextSystem, predict_pick
from utils import parse_args


prefix = os.getenv("CLUSTER_ROUTE_PREFIX", "").rstrip("/")

args = parse_args()
text_sys = TextSystem(args)

app = FastAPI(
    title="Do_an",
    version="1.0",
    description="check",
    openapi_prefix=prefix,
)

@app.get("/", include_in_schema=False)
def docs_redirect():
    return RedirectResponse(f"{prefix}/docs")

async def file_to_image(file):
    """Parser image in bytes"""
    npimg = np.frombuffer(await file.read(), np.int8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    return img

async def base64_to_image(img_str):
    imgdata = base64.b64decode(str(img_str))
    img = Image.open(io.BytesIO(imgdata))
    opencv_img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
    return opencv_img

# @app.post("/api/v1")
# async def process_img(files: List[UploadFile] = File(...)):
#     for file in files:
#         image = await file_to_image(file)
#         print(image.shape)
        
        
#         return img
def predict_kie(image):
    res = text_sys.mapping(image)
    with open(f'/home/tima/Do_an/infer/a.tsv', 'w', encoding='utf-8') as f:
            i = 1
            for r in res:
                texts = r["transcription"] 
                bboxs = r["points"]

                kie_save = str(str(i) + ',' + str(bboxs) + ',' +texts+'\n').replace("[","").replace("]","").replace(" ","")
                # print(kie_save)
                f.writelines(kie_save)
                i+=1
    entities = predict_pick(args)
    print(entities)
    item = []

    dict = {
        "time" : "null",
        "item" : [],
        "total": "null"
    }
    for entitie in entities:
        if entitie['entity_name'] == 'item':
            item.append(entitie['text'])
        elif entitie['entity_name'] == 'time':
            dict["time"] = entitie['text']
        elif entitie['entity_name'] == 'total':
            dict["total"] = entitie['text']
    dict["item"] = item
    return dict

@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    start = time.time()
    img = await file_to_image(file)
    img = scan_img(img)
    cv2.imwrite('/home/tima/Do_an/api/a.jpg',img)

    dict = predict_kie(img)
        

    print(time.time() - start)
    
    return JSONResponse(dict)

class ImageStr(BaseModel):
    image: str

async def url_to_image(img_str):
    import requests
    from io import BytesIO
    response = requests.get(img_str)
    img = Image.open(BytesIO(response.content)) 
    img = np.array(img)
    return img

@app.post("/api/predict_url")
async def predict(image: ImageStr):
    start = time.time()
    img = await url_to_image(image.image)
    img = scan_img(img)
    cv2.imwrite('/home/tima/Do_an/api/a.jpg',img)
    dict = predict_kie(img)
    print(time.time() - start)
    
    return JSONResponse(dict)


        
if __name__ == '__main__':
    uvicorn.run('api:app', port=15001)
from requests.api import head
from fastapi import FastAPI, Header, requests
from pydantic.main import BaseModel
from typing import Optional

import uvicorn
import json
import time
import os
#import logging
import requests
import numpy as np
import cv2

from google.cloud import storage
from fastapi.middleware.cors import CORSMiddleware

import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

from points_convert_test  import pts_convert_test
from calib import Calibration

class FromFrontendRequest(BaseModel):
    group_id: int
    project_id: int
    obj: dict
    file_name: str
    view_type: str
    up_axis: str
    
app = FastAPI()
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins = origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

    

@app.get("/")
async def root():
    return {"message" : "3D_pts to 2D_pts server page"}

@app.post(path='/result')
async def result(request: FromFrontendRequest,referer: Optional[str] = Header(None, convert_underscores=False)):
          
    start_time = time.time()
    group_id = request.group_id
    project_id = request.project_id
    obj = request.obj
    file_name = request.file_name
    view_type = request.view_type
    up_axis = request.up_axis
   
    #
    
    name = file_name.split('.')[-2]
    
    os.makedirs('./static/'+name, exist_ok=True)
    image_plan={}
    storage_client = storage.Client.from_service_account_json('...json')
    bucket = storage_client.get_bucket("pcd_calib")
    gcs_link = str(group_id) + '/' + str(project_id) + '/' + 'calib' + '/' + name + '/' + str(view_type) + '.txt'
    print(gcs_link)
    blobs = bucket.blob(gcs_link)    
    blobs.download_to_filename('./static/'+name + '.txt')
    
    calib_path = './static/'+name + '.txt'
    w = 2000
    h = 2000

    json_data = pts_convert_test(obj, name, calib_path, up_axis,w,h)
         
    os.remove('./static/'+name+'.txt')
    
    image_pts = jsonable_encoder(json_data)
    print(image_pts)
    
    return JSONResponse(content=image_pts)
    
if __name__ == '__main__':
    uvicorn.run("gateway:app",
                host="0.0.0.0",
                port=8182,
                reload=True,
                workers=1,
                #ssl_keyfile="./....pem",  
                #ssl_certfile="./....crt"
                )

from requests.api import head
from fastapi import FastAPI, Header, requests
from pydantic.main import BaseModel
from typing import Optional

import uvicorn
import json
import time
#import cv2
import os
import logging
#import requests
#import io
import numpy as np

from google.cloud import storage
from fastapi.middleware.cors import CORSMiddleware

import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from zipfile import ZipFile
#print(sys.path)

from loader import uploader

class FromFrontendRequest(BaseModel):
    gcs_link: str
    mode: str
    
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
    return {"message" : "3d data upload page"}

@app.post(path='/result')
async def result(request: FromFrontendRequest,referer: Optional[str] = Header(None, convert_underscores=False)):
          
    #revise.. the request.gcs_link does not exist in request....
    #divide upload api and return image list api..   
    #in retrun image list api, bucket(?) create / remove/ move file/ get storage name is needed
    #get storage name from bucket (image & calib)
    #create bucket using group_id / project_id /
    #move all file in old bucket to new bucket
    #remove old bucket
          
    start_time = time.time()
    gcs_link = request.gcs_link
    mode = request.mode   
          
    uploader(gcs_link, mode)
    
    return {'status': 'upload is done'}
    
if __name__ == '__main__':
    uvicorn.run("gateway:app",
                host="0.0.0.0",
                port=8080,
                reload=True,
                workers=1,
                )

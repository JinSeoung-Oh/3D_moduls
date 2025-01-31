import numpy as np
import open3d as o3d
import pickle
import os.path
import os
from datetime import datetime
import pandas as pd
from flask_restx import reqparse
import re
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor
from celery import Celery
import sys
print(sys.path)
print(os.getcwd())

from gcs_upload import upload_blob_calib, upload_blob_images, upload_blob_pcd, upload_blob_label
from util import extract_zip, convert_metadata_to_json, make_zip
import shutil
from zipfile import ZipFile
import json
import time
from pathlib import Path


def make_dataset_folder(directory):
    
    items = os.listdir(directory)
    items = [(os.path.join(directory, f)) for f in items]
    items = sorted(items)

    #print(f'Found {len(items)} folder imgs')

    return items

CELERY_BROKER_URL = 'redis://redis:8081'
CELERY_RESULT_BACKEND = 'redis://redis:8081'

papp = Celery("tasks", broker=CELERY_BROKER_URL,backend=CELERY_RESULT_BACKEND)

@papp.task(bind=True)                     
def uploader(self, gcs_link, mode):
    #args = parser()
    #timestr = time.strftime('%Y%m%d')
    name = gcs_link.split('/')[-1].split('.')[-2]
    name = name.split('_')
    input_file_path =name[0] + '/' + name[1]
    
    
    if not os.path.exists(input_file_path):
        os.makedirs(input_file_path)
    extract_zip(gcs_link, input_file_path)
    
    check = make_dataset_folder(input_file_path)
    check_name = check[0].split('/')[-1]
    fl = gcs_link.split('/')[-1].split('.')[-2]

    if mode == 'infer':
       if check_name == fl:
          #print(input_file_path)
          #print(check_name)
          check_path = input_file_path + '/' + fl + '/' +'calib'
          if os.path.exists(check_path) == True:
             shutil.move(input_file_path + '/' + fl + '/' +'calib', input_file_path)
             shutil.move(input_file_path + '/' + fl + '/' +'image_2', input_file_path)
             shutil.move(input_file_path + '/' + fl + '/' + 'pcd', input_file_path)
             shutil.rmtree(input_file_path + '/' + fl)
          else:
               shutil.move(input_file_path + '/' + fl + '/' +'image_2', input_file_path)
               shutil.move(input_file_path + '/' + fl + '/' + 'pcd', input_file_path)
               shutil.rmtree(input_file_path + '/' + fl)
       else:
           input_file_path = input_file_path
           
    elif mode == 'train' or mode == 'infer_bev':
        if check_name == fl:
           shutil.move(input_file_path + '/' + fl + '/' + 'calib', input_file_path)
           shutil.move(input_file_path + '/' + fl + '/' + 'image_2', input_file_path)
           shutil.move(input_file_path + '/' + fl + '/' + 'pcd', input_file_path)
           shutil.move(input_file_path + '/' + fl + '/' + 'label', input_file_path)
           shutil.rmtree(input_file_path + '/' + fl)
        else:
            input_file_path = input_file_path           
        
    if mode == 'train':
       label_file_list=[]
       label_path = input_file_path + '/' + 'label'
       label_files = make_dataset_folder(label_path)
       
       for path in label_files:
          if path.endswith('.DS_Store') or path.endswith('__MACOSX'):
             os.remove(path)
             
       label_files = make_dataset_folder(label_path)
       zip_path = label_path + '/label.zip'
       shutil.make_archive(label_path, 'zip', label_path)
       shutil.move(label_path + '.zip', label_path + '/label.zip')
       
       for j in range(len(label_files)):
           label_file = label_files[j]
           label_name = make_dataset_folder(label_file)
           label_file_list.append(label_name)
           
       label_file_list.append([zip_path])
       
       threadworkers=80//3
       with ThreadPoolExecutor(threadworkers) as pool:
          for s in range(len(label_file_list)):
             gcs_addresses=list(pool.map(upload_blob_label,label_file_list[s]))
             
    elif mode == 'infer_bev':
         label_file_list = []
         label_path = input_file_path + '/' + 'label'
         label_files = make_dataset_folder(label_path)
         
         for path in label_files:
            if path.endswith('.DS_Store') or path.endswith('__MACOSX'):
               os.remove(path)
               
         label_files = make_dataset_folder(label_path)
         zip_path = label_path + '/label.zip'
         shutil.make_archive(label_path, 'zip', label_path)
         shutil.move(label_path + '.zip', label_path + '/label.zip')     
         
         for path in label_files:
             label_file_list.append([path])
             
         label_file_list.append([zip_path])
         
         threadworkers=80//3
         with ThreadPoolExecutor(threadworkers) as pool:
            for s in range(len(label_file_list)):
               gcs_addresses=list(pool.map(upload_blob_label,label_file_list[s]))         
         
         
    #upload calib_file
    calib_file_list = []
    calib_path = input_file_path + '/' + 'calib'
    if os.path.exists(calib_path) == True:
       calib_files = make_dataset_folder(calib_path)
    
       for path in calib_files:
          if path.endswith('.DS_Store') or path.endswith('__MACOSX'):
             os.remove(path)
          
       calib_files = make_dataset_folder(calib_path)
       zip_path = calib_path + '/calib.zip'
       shutil.make_archive(calib_path, 'zip', calib_path)
       shutil.move(calib_path + '.zip', calib_path + '/calib.zip')
    
       for j in range(len(calib_files)):
          calib_file = calib_files[j]
          calib_name = make_dataset_folder(calib_file)
          calib_file_list.append(calib_name)
       
       calib_file_list.append([zip_path])
      
       threadworkers=80//3
       with ThreadPoolExecutor(threadworkers) as pool:
           for s in range(len(calib_file_list)):
              gcs_addresses=list(pool.map(upload_blob_calib,calib_file_list[s]))
        
    #upload image files
    image_file_list = []          
    image_path = input_file_path + '/' + 'image_2'
    image_files = make_dataset_folder(image_path)
    
    for path in image_files:
       if path.endswith('.DS_Store') or path.endswith('__MACOSX'):
          os.remove(path)
          
    image_files = make_dataset_folder(image_path)
    zip_path = image_path + '/image_2.zip'
    shutil.make_archive(image_path, 'zip', image_path)
    shutil.move(image_path +'.zip', image_path + '/image_2.zip')
                 
    for i in range(len(image_files)):
       image_file = image_files[i]
       image_name = make_dataset_folder(image_file)
       image_file_list.append(image_name)
              
    image_file_list.append([zip_path]) 
    threadworkers=80//3
    with ThreadPoolExecutor(threadworkers) as pool:
        for k in range(len(image_file_list)):
           gcs_addresses=list(pool.map(upload_blob_images,image_file_list[k]))        
    
    #upload pcd files
    #pcd_file_list = []          
    pcd_path = input_file_path + '/' + 'pcd'
    pcd_files = make_dataset_folder(pcd_path)
    
    for path in pcd_files:
       if path.endswith('.DS_Store') or path.endswith('__MACOSX'):
          os.remove(path)
          
    pcd_files = make_dataset_folder(pcd_path)    
    zip_path = pcd_path + '/pcd.zip'
    shutil.make_archive(pcd_path, 'zip', pcd_path)
    shutil.move(pcd_path +'.zip', pcd_path + '/pcd.zip')
    
    pcd_files.append(zip_path)           
    threadworkers=80//3
    #print(pcd_files)
    with ThreadPoolExecutor(threadworkers) as pool:
        for pcd_pa in pcd_files:
           pcd_ = [pcd_pa]
           gcs_addresses=list(pool.map(upload_blob_pcd,pcd_))                                          
    shutil.rmtree(input_file_path)  

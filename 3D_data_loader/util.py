from zipfile import ZipFile
import requests
import os
import glob
import json
from io import BytesIO
import shutil

def extract_zip(gcs_path, zip_path):
    zip_url = requests.get(gcs_path)
    zipfile = ZipFile(BytesIO(zip_url.content))
    zipfile.extractall(zip_path)
    zipfile.close()
    
    
def convert_metadata_to_json(metadata, files_dir):
    metadata_path=f"{files_dir}/metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata,f)
    return metadata_path
    
def make_zip(file_list, out_path):
    zip_file = ZipFile('result.zip', 'w')
    for f in file_list:
        zip_file.write(f)
    zip_file.close()
    out = out_path + '/'
    
    shutil.move('./result.zip', out)

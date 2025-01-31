from google.cloud import storage
import pandas as pd
import requests
import datetime
from pytz import timezone

storage_client = storage.Client.from_service_account_json('mykey.json')
adapter = requests.adapters.HTTPAdapter(pool_connections=200, pool_maxsize=200, max_retries=5)
storage_client._http.mount("https://", adapter)
storage_client._http._auth_request.session.mount("https://", adapter)

def upload_blob_calib(file_path):
    bucket_name='pcd_calib'
    bucket = storage_client.bucket(bucket_name)
    destination_blob_name = f"{file_path}"
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(file_path)
    
    return f"https://storage.googleapis.com/{bucket_name}/{destination_blob_name}"
    
def upload_blob_images(file_path):
    bucket_name='pcd_2d'
    bucket = storage_client.bucket(bucket_name)
    destination_blob_name = f"{file_path}"
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(file_path)
    
    return f"https://storage.googleapis.com/{bucket_name}/{destination_blob_name}"    
    
def upload_blob_pcd(file_path):
    bucket_name='pcdstorage'
    bucket = storage_client.bucket(bucket_name)
    print(file_path)
    destination_blob_name = f"{file_path}"
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(file_path)
    
    return f"https://storage.googleapis.com/{bucket_name}/{destination_blob_name}"
    
def upload_blob_label(file_path):
    bucket_name='pcd_label'
    bucket = storage_client.bucket(bucket_name)
    print(file_path)
    destination_blob_name = f"{file_path}"
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(file_path)
    
    return f"https://storage.googleapis.com/{bucket_name}/{destination_blob_name}"


def convert_gcs_address_to_csv(data_type,addresses, file_dir):
    today_data=datetime.datetime.now(timezone('Asia/Seoul'))
    short_data=today_data.strftime('%Y-%m-%d')
    df_name = f"{file_dir}/{data_type}_{short_data}.csv"
    if len(addresses)>0:
        info_function=lambda x: x.split("/")[-1].split(".")[0].split('_')
        file_info = [{"original_name": "_".join(info_function(address)[:-1]),
                      "download_address":address} for address in addresses]
    df = pd.DataFrame(file_info)
    df.sort_values(['original_name'], ignore_index=True, inplace=True)
    df.to_csv(df_name)
    
    return df_name

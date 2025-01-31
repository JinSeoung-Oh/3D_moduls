import os
from google.cloud.sql.connector import Connector
import sqlalchemy
from sqlalchemy import MetaData, update
from sqlalchemy.orm import Session
import datetime
from pytz import timezone

#parser = argparse.ArgumentParser()
#parser.add_argument("--service_type", required=True, type=str, default=None)
#args = parser.parse_args()
#service_type = args.service_type

os.environ["GOOGLE_APPLICATION_CREDENTIALS"]='./...json'

connector = Connector()

def getconn():
    conn = connector.connect("db_name",
                             "pymysql",
                             user="root",
                             password="mldnjrtm",
                             db="mlworks")
    return conn

pool = sqlalchemy.create_engine("mysql+pymysql://", creator=getconn)

#table_names = pool.table_names()
def db_update(request_id, address):
    today_data = datetime.datetime.now(timezone('Asia/Seoul'))
    short_data = today_data.strftime('%Y-%m-%d')
    long_data = today_data.strftime('%Y-%m-%d_%H:%M:%S')
    meta=MetaData()
    table_name = 'pcd_convert_result_table'
    meta.reflect(pool)
    
    with Session(pool) as db_conn:
       Table=meta.tables[table_name]
       u=update(Table)
       u=u.where(Table.c.request_status=='처리중', Table.c.request_id==int(request_id))
       u=u.values({'request_status':'완료',
                   'gcs_result_address':address,
                   'done_datatime':long_data})
                   
       db_conn.execute(u)
       db_conn.commit()
       
        


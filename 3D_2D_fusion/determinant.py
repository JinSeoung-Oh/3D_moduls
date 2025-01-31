import math
import os
import sys

import cv2
import numpy as np
import torch
import json
import numpy as np
import math

import pandas as pd
from calib import Calibration
from celery import Celery


def make_dataset_folder(directory):
    
    items = os.listdir(directory)
    items = [(os.path.join(directory, f)) for f in items]
    items = sorted(items)

    #print(f'Found {len(items)} folder imgs')

    return items


def get_corners(x, y, w, l, yaw):
    bev_corners = np.zeros((4, 2), dtype=np.float32)
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    # front left
    bev_corners[0, 0] = x - w / 2 * cos_yaw - l / 2 * sin_yaw
    bev_corners[0, 1] = y - w / 2 * sin_yaw + l / 2 * cos_yaw

    # rear left
    bev_corners[1, 0] = x - w / 2 * cos_yaw + l / 2 * sin_yaw
    bev_corners[1, 1] = y - w / 2 * sin_yaw - l / 2 * cos_yaw

    # rear right
    bev_corners[2, 0] = x + w / 2 * cos_yaw + l / 2 * sin_yaw
    bev_corners[2, 1] = y + w / 2 * sin_yaw - l / 2 * cos_yaw

    # front right
    bev_corners[3, 0] = x + w / 2 * cos_yaw - l / 2 * sin_yaw
    bev_corners[3, 1] = y + w / 2 * sin_yaw + l / 2 * cos_yaw

    return bev_corners

def get_2d_pts(labels, calib):
    label = labels[0]
    infor = []
    location, dim, ry = label[0:3], label[3:6], label[6]
    #print('loc',location)
    #print('dim',dim)
    #print('angle',ry)
    corners_3d = compute_box_3d(dim, location, ry)
    #print('3D coordination', corners_3d)
    corners_2d = project_to_image(corners_3d, calib.P2)
    #print('2D coordination', corners_2d)
    corners_2d = corners_2d.tolist()
    information = [corners_2d]
    infor.append(information)
        #img = draw_box_3d(img, corners_2d, color=[255,0,0])

    #return img
    return infor

def convert_real_to_point(point, boundary, info, bev):
    bev_height = bev[1]
    bev_width = bev[0]
    detection=[]
    for cla,x,y,z,h,w,l,yaw in point:
        _yaw = -yaw
        _y = bev_height*(x-boundary['minX']) / info[0]	
        _x = bev_width*(y-boundary['minY']) / info[1]
        _z = z-boundary['minZ']
        _h = h
        _w = w*(bev_width) / info[1]
        _l = l*(bev_height) / info[0]
        
        detection.append([cla, _x, _y, _z, _h, _w, _l, _yaw])
               
    return np.array(detection)

def drawRotatedBox(img, x, y, w, l, yaw, color):
    bev_corners = get_corners(x, y, w, l, yaw)
    corners_int = bev_corners.reshape(-1, 1, 2).astype(int)
    cv2.polylines(img, [corners_int], True, color, 2)
    corners_int = bev_corners.reshape(-1, 2)
    #corners_int = int(corners_int)
    #print('??', corners_int)
    cv2.line(img, (int(corners_int[0, 0]), int(corners_int[0, 1])), (int(corners_int[3, 0]), int(corners_int[3, 1])), (255, 255, 0), 2)
    

def drawpoint(img,x,y):
    x = int(x)
    y = int(y)
    cv2.circle(img, (x,y), 1, (255,0,0), 3)
    #cv2.circle(img, (0,0), 1, (255,0,0), 3)
    
    
def draw_predictions(img, detections,colors):
    for j in range(len(detections)):
        #print(j)
        if len(detections[j]) > 0:
            for det in detections:
                #(x-1:2, y-2:3, z-3:4, dim-4:7, yaw-7:8)
                #print(det)
                _cls,_x, _y, _z, _h, _w, _l, _yaw = det
                drawRotatedBox(img, _x, _y, _w, _l, _yaw, colors[int(j)])
                drawpoint(img, _x,_y)
    
    #cv2.imshow('check', img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    
    return img

def roty(angle):
    # Rotation about the y-axis.
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]])


def compute_box_3d(dim, location, ry):
    # dim: 3
    # location: 3
    # ry: 1
    # return: 8 x 3
    R = roty(ry)
    h, w, l = dim
    #print('ro',R)
    #print(h,w,l)
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

    corners = np.array([x_corners, y_corners, z_corners], dtype=np.float32)
    corners_3d = np.dot(R, corners)
    corners_3d = corners_3d + np.array(location, dtype=np.float32).reshape(3, 1)
    
    return corners_3d.transpose(1, 0)

def convert_list_to_array(info_list):
    info = []
    #print(info_list)
    cls_id =np.array(info_list[0])
    x = np.array(info_list[1], dtype=np.float)
    y = np.array(info_list[2], dtype=np.float)
    z = np.array(info_list[3], dtype=np.float)
    h = np.array(info_list[4], dtype=np.float)
    w = np.array(info_list[5], dtype=np.float)
    l = np.array(info_list[6], dtype=np.float)
    yaw = np.array(info_list[7], dtype=np.float)

    #print('cl',cls_id)
    #print('c_x',x)
    #print('c_y',y)
    #print('c_z',z)
    #print('h',h)
    #print('w',w)
    #print('l',l)
    #print('yaw',yaw)        
    info.append([cls_id, x,y,z,h,w,l,yaw])
        
    return np.array(info)
    
def convert_det_to_real_values(detections):
    kitti_dets = []
    boundary = {
    "minX": 0,
    "maxX": 50,
    "minY": -25,
    "maxY": 25,
    "minZ": -2.73,
    "maxZ": 1.27
     }

    bound_size_x = boundary['maxX'] - boundary['minX']
    bound_size_y = boundary['maxY'] - boundary['minY']
    bound_size_z = boundary['maxZ'] - boundary['minZ']

    boundary_back = {
    "minX": -50,
    "maxX": 0,
    "minY": -25,
    "maxY": 25,
    "minZ": -2.73,
    "maxZ": 1.27
    }

    BEV_WIDTH = 608  # across y axis -25m ~ 25m
    BEV_HEIGHT = 608  # across x axis 0m ~ 50m
    for det in detections:
       # (scores-0:1, x-1:2, y-2:3, z-3:4, dim-4:7, yaw-7:8)
        det = det.astype(float)
        _x, _y, _z, _h, _w, _l, _yaw = det
        _yaw = -_yaw
        x = _y / BEV_HEIGHT * bound_size_x + boundary['minX']
        y = _x / BEV_WIDTH * bound_size_y + boundary['minY']
        z = _z + boundary['minZ']
        w = _w / BEV_WIDTH * bound_size_y
        l = _l / BEV_HEIGHT * bound_size_x

        kitti_dets.append([x, y, z, _h, w, l, _yaw])

    return np.array(kitti_dets)    
  
        
def lidar_to_camera_box(boxes,V2C=None, R0=None, P2=None):
    ret = []
    for box in boxes:
        #print('ry', box[7])
        x,y,z,h,w,l,ry = np.array(box[1], dtype=np.float), np.array(box[2], dtype=np.float), np.array(box[3], dtype=np.float), np.array(box[4], dtype=np.float), np.array(box[5], dtype=np.float), np.array(box[6], dtype=np.float), np.array(box[7], dtype=np.float)
        (x,y,z),h,w,l,ry = lidar_to_camera(x,y,z,V2C=V2C, R0=R0, P2=P2), h,w,l,-ry - np.pi / 2
        ret.append([x,y,z,h,w,l,ry])
        
    return np.array(ret).reshape(-1,7)
    
    
def lidar_to_camera(x,y,z, V2C=None, R0 = None, P2 = None):
    x = np.array(x, dtype=np.float)
    y = np.array(y, dtype=np.float)
    z = np.array(z, dtype=np.float)
    p = np.array([x,y,z,1])
    if V2C is None or R0 is None:
        p = np.matmul(Tr_velo_to_cam, p)
        p = np.matmul(R0, p)
    else:
        p = np.matmul(V2C, p)
        p = np.matmul(R0, p)
    p = p[0:3]
    
    return tuple(p)

def project_to_image(pts_3d, P):
    # pts_3d: n x 3
    # P: 3 x 4
    # return: n x 2
    pts_3d_homo = np.concatenate([pts_3d, np.ones((pts_3d.shape[0], 1), dtype=np.float32)], axis=1)
    pts_2d = np.dot(P, pts_3d_homo.transpose(1, 0)).transpose(1, 0)
    pts_2d = pts_2d[:, :2] / pts_2d[:, 2:]

    return pts_2d.astype(np.int64)

def draw_box_3d(image, corners, color=(0, 0, 255)):
    face_idx = [[0, 1, 5, 4],
                [1, 2, 6, 5],
                [2, 3, 7, 6],
                [3, 0, 4, 7]]
    for ind_f in range(3, -1, -1):
        f = face_idx[ind_f]
        for j in range(4):
            cv2.line(image, (corners[f[j], 0], corners[f[j], 1]),
                     (corners[f[(j + 1) % 4], 0], corners[f[(j + 1) % 4], 1]), color, 2, lineType=cv2.LINE_AA)
        if ind_f == 0:
            cv2.line(image, (corners[f[0], 0], corners[f[0], 1]),
                     (corners[f[2], 0], corners[f[2], 1]), color, 1, lineType=cv2.LINE_AA)
            cv2.line(image, (corners[f[1], 0], corners[f[1], 1]),
                     (corners[f[3], 0], corners[f[3], 1]), color, 1, lineType=cv2.LINE_AA)

    return image
    
def get_yaw(Rotation):
    rotation_matrix = np.array(Rotation)
    r_mat = rotation_matrix.reshape(3,3)
    
    yaw = math.atan2(r_mat[2,1], r_mat[1,1,])
    pitch = math.atan2(-r_mat[3,1], math.sqrt(r_mat[3,2]**2 + r_mat[3,3]**2))
    roll = math.atan2(r_mat[3,2], r_mat[3,3])
    
    return yaw, pitch, roll
    
    
CELERY_BROKER_URL = 'redis://redis:8081'
CELERY_RESULT_BACKEND = 'redis://redis:8081'

papp = Celery("tasks", broker=CELERY_BROKER_URL,backend=CELERY_RESULT_BACKEND)

@papp.task(bind=True) 
def deter_area(self,info, name, calib, up_axis, w):
    data = info
    calib_info = calib
    
    
    result = {}
    info = [name,data["position"]["x"], data["position"]["y"], data["position"]["z"], data["scale"]["z"], data["scale"]["y"],data["scale"]["x"]]
    calib = Calibration(calib_info)
    rotation = calib.R0
    #print('in json', info)
    #print('roatat', rotation)
       
    yaw = data["rotation"]["z"]
    if up_axis == 'z':
      if yaw == 0:
          yaw, pitch, roll = get_yaw(rotation)
          info.append(yaw)
      else:
          if yaw == 0:
              yaw, pitch, roll = get_yaw(rotation)
              info.append(yaw)
          else:
              rot = -yaw - (math.pi/2)
              info.append(yaw)
    info = convert_list_to_array(info)
    info = lidar_to_camera_box(info, calib.V2C, calib.R0, calib.P2)
    informa = get_2d_pts(info, calib)
    information = informa[0][0]
    result["annotation"] = "CUBOID"
    result["front"] = {"coords":{"tl":{"x":information[4][0],"y":information[4][1]},"tr":{"x":information[6][0],"y":information[6][1]},"bl":{"x":information[3][0],"y":information[3][1]},"br":{"x":information[2][0],"y":information[2][1]}},"top":information[4][1],"left":information[4][0],"width":abs(information[4][0]-information[6][0]),"height":abs(information[4][1]-information[3][1])}
    result["back"] = {"coords":{"tl":{"x":information[7][0],"y":information[7][1]},"tr":{"x":information[5][0],"y":information[5][1]},"bl":{"x":information[0][0],"y":information[0][1]},"br":{"x":information[1][0],"y":information[1][1]}},"top":information[7][1],"left":information[7][0],"width":abs(information[7][0]-information[5][0]),"height":abs(information[7][1]-information[0][1])}  
        
    flag = True
        
    return result, flag
       

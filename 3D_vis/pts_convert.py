import math
import os
import sys
import argparse

import cv2
import numpy as np
import torch
import json
import struct

import pandas as pd
from plyfile import PlyData
#import open3d as o3d
from calib import Calibration
import time

def parser():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--out_path', type=str, default=None, help='enter your save path')
    parser.add_argument('--json_dir',  type=str, help='Enter your json file directory. ex) ./result/json')
    parser.add_argument('--img_path', type=str, help = 'Enter your image file directory. ex) ./data/image')
    parser.add_argument('--calib_path', type=str, help = 'Enter your calib file directory. ex) ./data/calib')
    
    args = parser.parse_args()
    
    return args

def make_dataset_folder(directory):
    
    items = os.listdir(directory)
    items = [(os.path.join(directory, f)) for f in items]
    items = sorted(items)

    #print(f'Found {len(items)} folder imgs')

    return items

def bin_to_pcd(filepath, out_path,img_id):
    out_path = out_path + str(img_id) + '.pcd'
    size_float = 4
    list_pcd = []
    with open(filepath, "rb") as f:
        byte = f.read(size_float * 4)
        while byte:
            x,y,z,intensity = struct.unpack("ffff", byte)
            list_pcd.append([x,y,z])
            byte = f.read(size_float * 4)
    np_pcd = np.asarray(list_pcd)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np_pcd)
    
    o3d.io.write_point_cloud(out_path, pcd)
    
    print('bin_to_pcd is done')

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

def get_2d_pts(img, labels, calib):
    #print(labels)
    infor = []
    for box_idx, label in enumerate(labels):
        cls_id, location, dim, ry = label[0], label[1:4], label[4:7], label[7]
        if location[2] < 2.0:  # The object is too close to the camera, ignore it during visualization
            print(location)
            continue
        if cls_id < 0:
            continue
        corners_3d = compute_box_3d(dim, location, ry)
        corners_2d = project_to_image(corners_3d, calib.P2)
        corners_2d = corners_2d.tolist()
        information = [cls_id, corners_2d]
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
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

    corners = np.array([x_corners, y_corners, z_corners], dtype=np.float32)
    corners_3d = np.dot(R, corners)
    corners_3d = corners_3d + np.array(location, dtype=np.float32).reshape(3, 1)
    return corners_3d.transpose(1, 0)

def convert_list_to_array(info_list):
    info = []
    for cls_id,x,y,z,h,w,l,yaw in info_list:
        info.append([cls_id, x,y,z,h,w,l,yaw])
        
    return np.array(info)
        
def lidar_to_camera_box(boxes,V2C=None, R0=None, P2=None):
    ret = []
    for box in boxes:
        x,y,z,h,w,l,ry = box
        (x,y,z),h,w,l,ry = lidar_to_camera(x,y,z,V2C=V2C, R0=R0, P2=P2), h,w,l,-ry - np.pi / 2
        ret.append([x,y,z,h,w,l,ry])
        
    return np.array(ret).reshape(-1,7)
    
    
def lidar_to_camera(x,y,z, V2C=None, R0 = None, P2 = None):
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

def main():
    args = parser()
    save_path = args.out_path
    json_dir = args.json_dir
    img_path = args.img_path
    calib_path = args.calib_path
    
    json_file = make_dataset_folder(json_dir)
    img_paths = make_dataset_folder(img_path)
    calib_paths = make_dataset_folder(calib_path)
    
    cor_info = {}
    for i in range(len(json_file)):
        with open(json_file[i], 'r') as f:
            data = json.load(f)
        info = data['dim_in_cloud']
        file_name = data['id'][0]
        file = file_name.split('/')[-1].split('.')[-2]
        #print(file)
        #project on image
        calib_pa = calib_paths[i]
        img = cv2.imread(img_paths[i])
        calib = Calibration(calib_pa)
        info = convert_list_to_array(info)
        info[:, 1:] = lidar_to_camera_box(info[:, 1:], calib.V2C, calib.R0, calib.P2)
        information = get_2d_pts(img, info, calib)
        cor_info[file_name] = information
        
        #print(type(cor_info))
        
        directory = save_path + '/' + file
        save = directory + '/' + 'result.json'
        if os.path.exists(directory) == False:
            os.mkdir(directory)
            
        with open(save, 'w') as f:
            json.dump(cor_info, f)
            
                         
if __name__ == "__main__":
    st = time.time()
    main()
    print(time.time()-st)

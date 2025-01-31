import math
import os
import sys
import argparse

import cv2
import numpy as np
import torch
import json

import pandas as pd
from plyfile import PlyData
import open3d as o3d

def parser():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--filepath', type=str, default=None, help='If KITTI like dataset, the file extension must be .bin(x,y,z,intensity). If data have (x,y,z,r,g,b), then the file extension must be .ply')
    parser.add_argument('--out_path', type=str, default=None, help='enter your save path')
    parser.add_argument('--bev_type', type=str, default='kitti', help="'kitti' or 'rgb'")
    parser.add_argument('--trans', type=str, default='True', help="if result image not correct, enter 'False'") 
    parser.add_argument('--json_dir',  type=str, help='Enter your json file directory. ex) ./result/json')
    
    args = parser.parse_args()
    
    return args

def make_dataset_folder(directory):
    
    items = os.listdir(directory)
    items = [(os.path.join(directory, f)) for f in items]
    items = sorted(items)

    #print(f'Found {len(items)} folder imgs')

    return items

def makeBEVMap(PointCloud_, boundary):
    BEV_WIDTH = 608  # across y axis -50m ~ 50m
    BEV_HEIGHT = 608  # across x axis -50m ~ 50m
    DISCRETIZATION = (boundary["maxX"] - boundary["minX"]) / BEV_HEIGHT
    Height = BEV_HEIGHT + 1
    Width = BEV_WIDTH + 1

    # Discretize Feature Map
    PointCloud = np.copy(PointCloud_)
    PointCloud[:, 0] = np.int_(np.floor(PointCloud[:, 0] / DISCRETIZATION))
    PointCloud[:, 1] = np.int_(np.floor(PointCloud[:, 1] / DISCRETIZATION) + Width / 2)

    # sort-3times
    sorted_indices = np.lexsort((-PointCloud[:, 2], PointCloud[:, 1], PointCloud[:, 0]))
    PointCloud = PointCloud[sorted_indices]
    _, unique_indices, unique_counts = np.unique(PointCloud[:, 0:2], axis=0, return_index=True, return_counts=True)
    PointCloud_top = PointCloud[unique_indices]

    # Height Map, Intensity Map & Density Map
    heightMap = np.zeros((Height, Width))
    intensityMap = np.zeros((Height, Width))
    densityMap = np.zeros((Height, Width))

    # some important problem is image coordinate is (y,x), not (x,y)
    max_height = float(np.abs(boundary['maxZ'] - boundary['minZ']))
    heightMap[np.int_(PointCloud_top[:, 0]), np.int_(PointCloud_top[:, 1])] = PointCloud_top[:, 2] / max_height

    normalizedCounts = np.minimum(1.0, np.log(unique_counts + 1) / np.log(64))
    intensityMap[np.int_(PointCloud_top[:, 0]), np.int_(PointCloud_top[:, 1])] = PointCloud_top[:, 3]
    densityMap[np.int_(PointCloud_top[:, 0]), np.int_(PointCloud_top[:, 1])] = normalizedCounts

    RGB_Map = np.zeros((3, Height - 1, Width - 1))
    RGB_Map[2, :, :] = densityMap[:BEV_HEIGHT, :BEV_WIDTH]  # r_map
    RGB_Map[1, :, :] = heightMap[:BEV_HEIGHT, :BEV_WIDTH]  # g_map
    RGB_Map[0, :, :] = intensityMap[:BEV_HEIGHT, :BEV_WIDTH]  # b_map

    return RGB_Map


def get_lidar(path, dim):
    lidar= np.fromfile(path, dtype=np.float32).reshape(-1,dim)
    return lidar

def get_filtered_lidar(lidar, boundary):
    minX = boundary['minX']
    maxX = boundary['maxX']
    minY = boundary['minY']
    maxY = boundary['maxY']
    minZ = boundary['minZ']
    maxZ = boundary['maxZ']
    
    mask = np.where((lidar[:,0]>=minX)&(lidar[:,0]<=maxX)&(lidar[:,1]>=minY)&(lidar[:,1]<=maxY)&(lidar[:,2]>=minZ)&(lidar[:,2]<=maxZ))
    lidar = lidar[mask]
    lidar[:,2] = lidar[:,2]-minZ
    
    return lidar

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


def drawRotatedBox(img, x, y, w, l, yaw, color):
    bev_corners = get_corners(x, y, w, l, yaw)
    corners_int = bev_corners.reshape(-1, 1, 2).astype(int)
    cv2.polylines(img, [corners_int], True, color, 2)
    corners_int = bev_corners.reshape(-1, 2)
    #corners_int = int(corners_int)
    print('??', corners_int)
    cv2.line(img, (int(corners_int[0, 0]), int(corners_int[0, 1])), (int(corners_int[3, 0]), int(corners_int[3, 1])), (255, 255, 0), 2)
    
def drawpoint(img,x,y):
    x = int(x)
    y = int(y)
    cv2.circle(img, (x,y), 1, (255,255,0), 3)
    #cv2.circle(img, (0,0), 1, (255,0,0), 3)
    
    
def draw_predictions(img, detections,colors):
    for j in range(len(detections)):
        if len(detections[j]) > 0:
            for det in detections:
                #(x-1:2, y-2:3, z-3:4, dim-4:7, yaw-7:8)
                #print(det)
                _cls,_x, _y, _z, _h, _w, _l, _yaw = det
                drawRotatedBox(img, _x, _y, _w, _l, _yaw, colors[int(j)])
                #drawpoint(img,'0', '0')

    return img

def main():
    args = parser()
    lidar_path = args.filepath
    save_path = args.out_path
    bev_type = args.bev_type
    trans = args.trans
    json_dir = args.json_dir
    
    if bev_type == 'kitti':
        boundary = {
            "minX": -50,
            "maxX": 50,
            "minY": -50,
            "maxY": 50,
            "minZ": -2.73,
            "maxZ": 1.27
        }

        bound_size_x = boundary['maxX'] - boundary['minX']   
        bound_size_y = boundary['maxY'] - boundary['minY']
        bound_size_z = boundary['maxZ'] - boundary['minZ']

        boundary_back = {
            "minX": -50,
            "maxX": 50,
            "minY": -50,
            "maxY": 50,
            "minZ": -2.73,
            "maxZ": 1.27
        }
        colors = [[0, 255, 255], [0, 0, 255], [255, 0, 0], [255, 120, 0],
          [255, 120, 120], [0, 120, 0], [120, 255, 255], [120, 0, 255]]
        
        lidar_paths = make_dataset_folder(lidar_path)
        json_file = make_dataset_folder(json_dir)
        for i in range(len(lidar_paths)):
            img_id = lidar_paths[i].split('/')[-1].split('.')[-2]
            lidar = get_lidar(lidar_paths[i], 4)
            lidar = get_filtered_lidar(lidar, boundary)
            bev_map = makeBEVMap(lidar, boundary)
            bev_map = torch.from_numpy(bev_map)
            bev_map = (bev_map.squeeze().permute(1,2,0).numpy()*255).astype(np.uint8)
            bev_map = cv2.resize(bev_map, (608,608))
            bev_map = cv2.rotate(bev_map, cv2.ROTATE_180)
            
            with open(json_file[i], 'r') as f:
                data = json.load(f)
            info = data['dim_in_cloud']
            bev_map = draw_predictions(bev_map, info,colors)
           
            save_path = save_path + '/' + str(img_id) + '.png'
            if trans == 'True':
                split_Y = np.split(bev_map,2,axis=0)
                rear = split_Y[0]
                front = split_Y[1]
                adjust_bev = np.concatenate((front, rear), axis=0)
                cv2.imwrite(save_path, adjust_bev)
           
            else:
                cv2.imwrite(save_path, bev_map)
            
    if bev_type == 'rgb':
        lidar_paths = make_dataset_folder(lidar_path)
        for i in range(len(lidar_paths)):
            img_id = lidar_paths[i].split('/')[-1].split('.')[-2]
            save_path = save_path + '/' + str(img_id) + '.png'
            plydata = PlyData.read(lidar_paths[i])
            data = plydata.elements[0].data
            data_pd = pd.DataFrame(data)
            data_np = np.zeros(data_pd.shape, dtype=np.float)
            property_names = data[0].dtype.names
            print('make info dataframe')
            for i, name in enumerate(property_names):
                data_np[:,i] = data_pd[name]
        
            datas=data_np.astype(np.float32)
            points = []
            colors = []
            print('gether pts & color')
            for x,y,z,r,g,b in datas:
                points.append([x,y,z])
                r = r/255
                g = g/255
                b = b/255
                colors.append([r,g,b])
            pts = np.array(points)
            pts = pts.reshape(-1,3)
            rgb = np.array(colors)
            rgb = rgb.reshape(-1,3)
            
            print('Vectorizing...')
            pc = o3d.geometry.PointCloud()
            pc.points = o3d.utility.Vector3dVector(pts)
            pc.colors = o3d.utility.Vector3dVector(rgb)
            
            print('generating vis bev map...')
            vis = o3d.visualization.Visualizer()
            vis.create_window()
            vis.add_geometry(pc)
            vis.get_render_option().point_size = 1.5 
            vis.run()
            vis.capture_screen_image(save_path, True)
            vis.destroy_window()
        
if __name__ == "__main__":
    main()

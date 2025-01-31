import math
import os
import sys
#import argparse

import cv2
import numpy as np
import torch
import json
import struct

import pandas as pd
from plyfile import PlyData
import open3d as o3d
from calib import Calibration
import gradio as gr

#def parser():
#    parser = argparse.ArgumentParser(description='arg parser')
#    parser.add_argument('--filepath', type=str, default=None, help='If KITTI like dataset, the file extension must be .bin(x,y,z,intensity). If data have (x,y,z,r,g,b), then the file extension must be .ply')
#    parser.add_argument('--out_path', type=str, default=None, help='enter your save path')
#    parser.add_argument('--bev_type', type=str, default='kitti', help="'kitti' or 'rgb'")
#    parser.add_argument('--trans', type=str, default='False', help="this command work for merge front and rare bev map") 
#    parser.add_argument('--json_dir',  type=str, help='Enter your json file directory. ex) ./result/json')
#    parser.add_argument('--img_path', type=str, help = 'Enter your image file directory. ex) ./data/image')
#    parser.add_argument('--calib_path', type=str, help = 'Enter your calib file directory. ex) ./data/calib')
    
#    args = parser.parse_args()
    
#    return args

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
    
    #print('bin_to_pcd is done')
    return out_path
 
def pcd_to_ply(filepath, out_path, img_id):
    out_path = out_path + str(img_id) + '.ply' 
    pcd = o3d.io.read_point_cloud(filepath)
    o3d.io.write_point_cloud(out_path, pcd)
    
    return out_path
       
def ply_to_obj(filepath, out_path, img_id):
    out_path = out_path + str(img_id) + '.obj'
    pcd = o3d.io.read_point_cloud(filepath)
    pts = np.array(pcd.points)
    
    edge_lenth = 0.1
    
    with open(out_path, 'a') as f:
        for i in range(len(pts)):
            n=i*8
            pt = pts[i]
            x = float(pt[0])
            y = float(pt[1])
            z = float(pt[2])
            
            f.write('v {0:.6f} {1:.6f} {2:.6f}\n'.format(x+edge_length, y+edge_length, z+edge_length))
            f.write('v {0:.6f} {1:.6f} {2:.6f}\n'.format(x-edge_length, y+edge_length, z+edge_length))
            f.write('v {0:.6f} {1:.6f} {2:.6f}\n'.format(x-edge_length, y-edge_length, z+edge_length))
            f.write('v {0:.6f} {1:.6f} {2:.6f}\n'.format(x+edge_length, y-edge_length, z+edge_length))
            f.write('v {0:.6f} {1:.6f} {2:.6f}\n'.format(x+edge_length, y+edge_length, z-edge_length))
            f.write('v {0:.6f} {1:.6f} {2:.6f}\n'.format(x-edge_length, y+edge_length, z-edge_length))
            f.write('v {0:.6f} {1:.6f} {2:.6f}\n'.format(x-edge_length, y-edge_length, z-edge_length))
            f.write('v {0:.6f} {1:.6f} {2:.6f}\n'.format(x+edge_length, y-edge_length, z-edge_length))

            f.write('f {0} {1} {2} {3}\n'.format(n+1, n+2, n+3, n+4))
            f.write('f {0} {1} {2} {3}\n'.format(n+5, n+6, n+7, n+8))
            f.write('f {0} {1} {2} {3}\n'.format(n+1, n+2, n+6, n+5))
            f.write('f {0} {1} {2} {3}\n'.format(n+2, n+3, n+7, n+6))
            f.write('f {0} {1} {2} {3}\n'.format(n+3, n+4, n+8, n+7))
            f.write('f {0} {1} {2} {3}\n'.format(n+4, n+1, n+5, n+8))
            
    return out_path

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

def show_rgb_image_with_boxes(img, labels, calib):
    for box_idx, label in enumerate(labels):
        cls_id, location, dim, ry = label[0], label[1:4], label[4:7], label[7]
        if location[2] < 2.0:  # The object is too close to the camera, ignore it during visualization
            print(location)
            continue
        if cls_id < 0:
            continue
        #print(dim)
        #print(location)
        #print(ry)
        corners_3d = compute_box_3d(dim, location, ry)
        #print(corners_3d)
        corners_2d = project_to_image(corners_3d, calib.P2)
        #print(corners_2d)
        img = draw_box_3d(img, corners_2d, color=[255,0,0])

    return img

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

    return pts_2d.astype(np.int)

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

#def main(args, obj_path):
def main(file_dir, json_dir, img_dir, calib_dir):
    #args = parser()
    #lidar_path = filepath
    #save_path = args.out_path
    #json_dir = json_path
    #img_path = img_path
    #calib_path = args.calib_path
    #json_dir = args.json_dir
    trans = 'False'
    bev_type = 'kitti'
    
    json_file = make_dataset_folder(json_dir)
    lidar_paths = make_dataset_folder(file_dir)
    img_paths = make_dataset_folder(img_dir)
    calib_paths = make_dataset_folder(calib_dir)
    
    
    #make bev_map
    if bev_type == 'kitti':
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
        colors = [[0, 255, 255], [0, 0, 255], [255, 0, 0], [255, 120, 0],
          [255, 120, 120], [0, 120, 0], [120, 255, 255], [120, 0, 255]]
        
        #for j in range(len(lidar_paths)):
        #    img_id = lidar_paths[j].split('/')[-1].split('.')[-2]
        #    path = bin_to_pcd(lidar_paths[j], './', img_id )
        #    p_path = pcd_to_ply(path, './', img_id)
        #    obj_path = ply_to_obj(p_path, './', img_id)
            
        #    os.remove(path)
        #    os.remove(p_path)
        
        for i in range(len(lidar_paths)):
            img_id = lidar_paths[i].split('/')[-1].split('.')[-2]
            lidar = get_lidar(lidar_paths[i], 4)
            lidar = get_filtered_lidar(lidar, boundary)
            bev_map = makeBEVMap(lidar, boundary)
            bev_map = torch.from_numpy(bev_map)
            bev_map = (bev_map.squeeze().permute(1,2,0).numpy()*255).astype(np.uint8)
            bev_map = cv2.resize(bev_map, (608,608))
            
            with open(json_file[i], 'r') as f:
                data = json.load(f)
            info = data['dim_in_cloud']
            bev = [608, 608]
            bound_size = [bound_size_x,bound_size_y, bound_size_z]
            info_1 = convert_real_to_point(info, boundary, bound_size, bev)
            bev_map = draw_predictions(bev_map, info_1,colors)
            bev_map = cv2.rotate(bev_map, cv2.ROTATE_180)
            
            #save_pa = save_path + str(img_id) + '_bev' + '.png'
            #cv2.imwrite(save_pa, bev_map)
    
            #project on image
            calib_pa = calib_paths[i]
            img = cv2.imread(img_paths[i])
            calib = Calibration(calib_pa)
            info = convert_list_to_array(info)
            info[:, 1:] = lidar_to_camera_box(info[:, 1:], calib.V2C, calib.R0, calib.P2)
            img_bgr = show_rgb_image_with_boxes(img, info, calib)
            #save_pat = save_path + str(img_id) +'_img' +'.png'
            #cv2.imwrite(save_pat, img_bgr)
            
            
            #draw_prediction_on_pcd
            #info = data['dim_in_cloud']
            #pcd_path = './'
            #bin_to_pcd(lidar_paths[i], pcd_path, img_id)
            #path = './' + img_id + '.pcd'
            #pcd = o3d.io.read_point_cloud(path)
            #pts = np.array(pcd.points)
            #colors = np.array(pcd.colors)
            #if len(colors) == 0:
            #    color = [0,0,0]
            #    colors = [] 
            #    for i in range(len(pts)):
            #        colors.append(color)
            #    colors = np.array(colors)
            #    colors = colors.reshape(-1, 3)
            #pc = o3d.geometry.PointCloud()
            #pc.points = o3d.utility.Vector3dVector(pts)
            #pc.colors = o3d.utility.Vector3dVector(colors)
            
            #convert info to coordination
            #poly_pts=[]
            #for box_id, infor in enumerate(info):
            #    cls_id, location, dim, ry = infor[0], infor[1:4], infor[4:7], infor[7]
            #    cor = compute_box_3d(dim, location, ry)
            #    information= [cls_id, cor]
            #    poly_pts.append(information)
            
            #define line_sets
           # k=0
           # line_sets=[]
           # for i in range(len(poly_pts)):
           #     k += 1
           #     information = poly_pts[i]
           #     points = information[1]
           #     lines = [[0,1],
           #              [1,2],
           #              [2,3],
           #              [3,0],
           #              [1,5],
           #              [2,6],
           #              [3,7],
           #              [0,4],
           #              [4,5],
           #              [5,6],
           #              [6,7],
           #              [7,4]]
           #     colors = [[1, 0, 0] for i in range(len(lines))]
           #     line_set = o3d.geometry.LineSet(
           #     points=o3d.utility.Vector3dVector(points),
           #     lines=o3d.utility.Vector2iVector(lines),
           #     )
           #     line_set.colors = o3d.utility.Vector3dVector(colors)
           #     globals()["line_set" + str(k)] = line_set
           #     line_sets.append(globals()["line_set" + str(k)])
            #save_path = save_path + str(img_id) +'_3D' +'.png'
            
           # if len(line_sets) == 0:
           #     pass
            
           # if len(line_sets) == 1:
           #     vis = o3d.visualization.Visualizer()
           #     vis.create_window()
           #     vis.add_geometry(pc)
           #     vis.add_geometry(line_set1)
           #     vis.get_render_option().point_size = 1.5 
           #     ctr = vis.get_view_control()
           #     ctr.set_zoom(0.3)
           #     vis.run()
           #     img = np.array(vis.capture_screen_float_buffer())
           #     vis.destroy_window()    
                
           # if len(line_sets) == 2:
           #     vis = o3d.visualization.Visualizer()
           #     vis.create_window()
           #     vis.add_geometry(pc)
           #     vis.add_geometry(line_set1)
           #     vis.add_geometry(line_set2)
           #     vis.get_render_option().point_size = 1.5 
           #     ctr = vis.get_view_control()
           #     ctr.set_zoom(0.3)
                #visible = False then vis.run is infinite loof, I guss
           #     vis.run()
           #     img = np.array(vis.capture_screen_float_buffer())
           #     vis.destroy_window()
                

                
    #print(vis)            
    return bev_map, img_bgr
                
if __name__ == "__main__":
    #args = parser()
    #input_lidar = args.filepath
    #obj_path = ply_to_obj(input_lidar, args.out_path)
    
    demo = gr.Interface(fn=main, inputs=["text","text", "text","text"], outputs=["image","image"])
    #main()
    demo.launch(share=True)

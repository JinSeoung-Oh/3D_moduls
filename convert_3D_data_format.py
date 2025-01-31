import numpy as np
import struct
import open3d as o3d
import pickle
import os.path
import os
from datetime import datetime
import pye57
import pandas as pd
#from plyfile import PlyData
import trimesh
import time
import pytest
import pylas
import velodyne_decoder as vd
import matplotlib.image as mpimg
import re
import shutil

#def parser():
#    parser = reqparse.RequestParser()
#    parser.add_argument('--rgbd_type', type=str, default=None, help='if your dataset is rgb-d, then enter this argument. If you do not want, enter None. Support format : Redwood, SUN RGB-D, NYU, TUM')
#    parser.add_argument('--pcap_type', type=str, default=None, help='if your data is velodyne pcap, then enthis this argument. If you do not want, enter None. Support format : Alpa Prime, HDL-32E, HDL-64E, Puck Hi-Res, Puck LITE, VLP-16, VLP-32A, VLP-32B, VLP-32C') 
    
#    args = parser.parse_args()
    
#    return args

def make_dataset_folder(directory):
    
    items = os.listdir(directory)
    items = [(os.path.join(directory, f)) for f in items]
    items = sorted(items)

    #print(f'Found {len(items)} folder imgs')

    return items

def bin_to_pcd(filepath, out_path, img_id, unvaild_path):
    out_path = out_path + '/' + str(img_id) + '.pcd'
    bin_pcd = np.fromfile(filepath, dtype=np.float32)
    if len(bin_pcd) % 4 == 0:
        points = bin_pcd.reshape((-1,4))[:, 0:3]
    
    elif len(bin_pcd) % 6 == 0:
        points = bin_pcd.reshape((-1,6))[:, 0:3]
        
    o3d_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    
    o3d.io.write_point_cloud(out_path, o3d_pcd)
    
    print('bin_to_pcd is done')


def write_ply(verts, colors, indices, out_path):
    if colors is None:
        colors = np.zeros_like(verts)
    if indices is None:
        indices = []

    file = open(out_path, 'w')
    file.write('ply \n')
    file.write('format ascii 1.0\n')
    file.write('element vertex {:d}\n'.format(len(verts)))
    file.write('property float x\n')
    file.write('property float y\n')
    file.write('property float z\n')
    file.write('property uchar red\n')
    file.write('property uchar green\n')
    file.write('property uchar blue\n')
    file.write('element face {:d}\n'.format(len(indices)))
    file.write('property list uchar uint vertex_indices\n')
    file.write('end_header\n')
    
    for vert, color in zip(verts, colors):
        file.write('{:f} {:f} {:f} {:d} {:d} {:d}\n'.format(vert[0], vert[1], vert[2], int(color[0]), int(color[1]), int(color[2])))
                                                           
    for ind in indices:
        file.write('3 {:d} {:d} {:d}\n'.format(ind[0], ind[1], ind[2]))

    file.close()
                    
    print('write_ply is done')
                   
def ply_to_pcd(input_path, output_path,img_id):
    #exten = output_path.split('/')[-1].split('.')[-1]
    #if exten == 'pcd':
    #    pcd = o3d.io.read_point_cloud(input_path)
    #    o3d.io.write_point_cloud(output_path, pcd)
    #else:
    output_path = output_path + '/' + str(img_id) + '.pcd'
    pcd = o3d.io.read_point_cloud(input_path)
    o3d.io.write_point_cloud(output_path, pcd)
    #os.remove(input_path)
    print('ply_to_pcd is done')
    
                   
def pcd_to_ply(input_path, output_path,img_id):
    #exten = output_path.split('/')[-1].split('.')[-1]
    #if exten == 'ply':    
    #    pcd = o3d.io.read_point_cloud(input_path)
    #    o3d.io.write_point_cloud(output_path,pcd)
    #else:
    output_path = output_path + '/' + str(img_id) + '.ply'
    pcd = o3d.io.read_point_cloud(input_path)
    o3d.io.write_point_cloud(output_path, pcd)
    print('pcd_to_ply is done')
                   
def process_line(line, transform):
    lin_info =line.rstrip().split(" ")
    point_1 = float(lin_info[0])
    point_2 = float(lin_info[1])
    point_3 = float(lin_info[2])
    add_dim = float('0')
    pts = [point_1, point_2, point_3, add_dim]
    vals = np.array(pts)
    pos = vals.dot(transform)
    return "{0} {1} {2}".format(pos[0], pos[1], pos[2])

def check_ptx(filename):
    if filename.endswith('.ptx'):
        return True
    return False

def convertor(root_dir):
    file_name_list=[]
    for (root, dirs, files) in os.walk(root_dir):
        if len(files) > 0:
            for file_name in files:
                if check_ptx(file_name):
                    file_name_list.append(file_name)      
    for i in range(len(file_name_list)):
        filename = file_name_list[i]
        print ("start: ", datetime.now(), filename)
        #print(root_dir)
        file_path = root_dir + '/' + filename
        name = filename.split('.')[-2]
        #print(file_path)
        with open(file_path, 'r') as ptx, open(root_dir + '/' +name + ".pcd", 'w') as pcd:
            lines = ptx.readlines()
            col = int(lines[0])
            row = int(lines[1])
            #print(np.array(lines[2].rstrip().split(" ")))
            #print(np.array(type(lines[2].rstrip().split(" "))))
            # transformation matrix for view point
            view_point_tm = np.zeros([4, 3])
            view_point_tm[3] = np.array(lines[2].rstrip().split(" "), dtype=np.float)
            view_point_tm[0] = np.array(lines[3].rstrip().split(" "), dtype=np.float)
            view_point_tm[1] = np.array(lines[4].rstrip().split(" "), dtype=np.float)
            view_point_tm[2] = np.array(lines[5].rstrip().split(" "), dtype=np.float)
            cloud_tm = np.zeros([4, 3])  # transformation matrix for cloud
            cloud_tm[0] = np.array(lines[6].rstrip().split(" ")[:3], dtype=np.float)
            cloud_tm[1] = np.array(lines[7].rstrip().split(" ")[:3], dtype=np.float)
            cloud_tm[2] = np.array(lines[8].rstrip().split(" ")[:3], dtype=np.float)
            cloud_tm[3] = np.array(lines[9].rstrip().split(" ")[:3], dtype=np.float)
           
            pcd_head = '''# .PCD v0.7 - Point Cloud Data file format
56VERSION 0.7
57FIELDS x y z
58SIZE 4 4 4
59TYPE F F F
60COUNT 1 1 1
61WIDTH {0}
62HEIGHT {1}extract_zip
63VIEWPOINT {2} {3} {4} 1 0 0 0
64POINTS {5}
65DATA ascii\n'''.format(col, row,
                       view_point_tm[3, 0],
                       view_point_tm[3, 1],
                       view_point_tm[3, 2],
                       col * row)
            pcd.writelines(pcd_head)
            line = lines[10:]
            print(len(line))
            for k in range(len(line)):
                pcd.write(process_line(line[k], cloud_tm) + "\n")
        print ("end  : ", datetime.now(), filename)
        
def e57_to_pcd(file_path, out_path, img_id):
    out_path = out_path + '/' + str(img_id) + '.ply'
    #print(file_path)
    e57 = pye57.E57(file_path)
    data = e57.read_scan_raw(0)

    key_list = []
    for k,v in data.items():
        key_list.append(k)
    
    if 'intensity' not in key_list:
        xs = data["cartesianX"]
        ys = data["cartesianY"]
        zs = data["cartesianZ"]
        rs = data["colorRed"]
        gs = data["colorGreen"]
        bs = data["colorBlue"]
        
        colors = []
        for i in range(len(xs)):
            color = [rs[i],gs[i],bs[i]]
            colors.append(color)
        text = None
    elif 'intensity' in key_list:
        xs = data["cartesianX"]
        ys = data["cartesianY"]
        zs = data["cartesianZ"]

        colors = None
        text = None
    #print("gether points..."
    verts = []
    for i in range(len(xs)):
        points = [xs[i], ys[i], zs[i]]
        verts.append(points)
    
    print("writting ply file...")
    write_ply(verts, colors, None, out_path)  
    #print("convertting ply to pcd...")
    #ply_to_pcd('./test_result_result.ply', out_path, img_id)
    #os.remove('./test_result_result.ply')
    #               
    return out_path
    
def convert_ply_to_bin(input_path, output_path, img_id):
    # _*_ coding: utf-8 _*_
    output_path = output_path + '/' + str(img_id) + '.bin'
    plydata = PlyData.read(input_path)
    data = plydata.elements[0].data
    data_pd = pd.DataFrame(data)
    data_np = np.zeros(data_pd.shape, dtype=np.float)
    property_names = data[0].dtype.names
    for i, name in enumerate(property_names):
        data_np[:, i] = data_pd[name]
    data_np.astype(np.float32).tofile(output_path)
                   
    print('ply_to_bin is done')
    
def obj_to_pcd(input_path, output_path, img_id):
    ply_path = output_path + '.ply'
    output_path = output_path + '/' + str(img_id) + '.pcd'
    
    vert = []
    file_in = open(input_path, 'r')
    while True:
        line = file_in.readline().strip()
        check = line.split(' ')
        if check[0] == 'v':
            vert.append(check[1:])
        if not line:break
    file_in.close()
    
    pts = []
    for k in range(len(vert)):
        pts_str = vert[k]
        x = float(pts_str[0])
        y = float(pts_str[1])
        z = float(pts_str[2])
        point = [x,y,z]
        pts.append(point)    
   
    colors=[]
    for z in range(len(pts)):
        color = [255,212,0]
        colors.append(color)
    
    write_ply(pts,colors, None, ply_path)    
    #ply_to_pcd(ply_path, output_path, img_id)
    #os.remove(ply_path)
                   
    #print('obj_to_pcd is done')
    return ply_path
    
def npypkl_to_pcd(input_path, output_path, img_id):
    with open(input_path, 'rb') as f:
        data = pickle.load(f) 
    v = torch.tensor(data)
    points = v[:, :3]
    color = [255, 255, 0]
    colors = []
    for i in range(0, points.shape[0]):
        colors.append(color)    
    write_ply(points, colors, None, './result.ply')
    #output_path = output_path + '/' + str(img_id) + '.pcd'
    #ply_to_pcd('./result.ply', output_path, img_id)
    #os.remove('./result.ply')
    path = './result.ply'
                   
    #print('np.array pkl file to pcd is done')
    return path


def las_to_pcd(input_path, output_path, img_id):
    output_path = output_path + '/' + str(img_id) + '.pcd'
    las = pylas.read(input_path)
    pts = np.vstack((las.X, las.Y, las.Z)).transpose()
    colors = []
    if len(las.red) != 0:
        for i in range(len(las.red)):
            r = las.red[i]/256
            g = las.green[i]/256
            b = las.blue[i]/256
            rgb = [r,g,b]
            colors.append(rgb)
    elif len(las.red) == 0:
       r = 255
       g = 255
       b = 0
       for i in range(len(las.X)):
           rgb = [r,g,b]
           colors.append(rgb)
    write_ply(pts, colors, None, './result.ply')
    #output_path = output_path + '/' + str(img_id) + '.pcd'
    #ply_to_pcd('./result.ply', output_path, img_id)
    #os.remove('./result.ply')
                   
    #print('las file to pcd is done')
    #pts = np.vstack((las.X, las.Y, las.Z)).transpose()
    path = './result.ply'

    return path
                   
def asc_to_pcd(filename):
   
    def convert(filename):
        print("the input file name is:%r." %filename)
        start = time.time()
        file = open(filename, "r+")
        count = 0
    
        for line in file:
            count = count+1
        print("size is %d" %count)
        file.close()
    
        f_prefix = filename.split('.')[0]
        output_file = '{perfix}.pcd'.format(prefix=f_prefix)
        output = open(output_file, 'w+')
    
        list_head = ['# .PCD v0.7 - Point Cloud Data file format\n', 'VERSION 0.7\n', 'FIELDS x y z intensity\n', 'SIZE 4 4 4 4\n', 'TYPE F F F F\n', 'COUNT 1 1 1 1\n', "VIEWPOINT 0 0 0 1 0 0  0\n"]
        output.writelines(list_head)
        output.write('WIDTH ')
        output.write(str(count))
        output.write('\HIGHT ')
        output.write(str(1))
        output.write('nPOINT ')
        output.write(str(count))
        output.write('\nDATA ascii\n')
    
        file1 = open(filename, "r")
        all = file1.read()
        output.write(all)
        output.close()
        file1.close()
    
        end = time.time()
        print("run time is:", end-start)

                    
def txt_to_pcd(input_path, output_path, img_id):
    output_path = output_path + '/' + str(img_id) + '.pcd'                 
    pts = []
    colors = []
    info = []
    with open(input_path, 'r') as f:
        data = f.readlines()
    for line in data:
        line = line.strip()
        line = line.split(' ')
        info.append(line)
    if len(info[0]) == 6:
        for i in range(len(info)):
            information = info[i]
            point = [float(information[0]), float(information[1]), float(information[2])]
            color = [float(information[3]), float(information[4]), float(information[5])]
            pts.append(point)
            colors.append(color)
        write_ply(pts, colors, None, './result.ply')
        path = './result.ply'
        #ply_to_pcd('./result.ply', output_path, img_id)
        #os.remove('./result.ply')
    else:
        color = [255, 255, 0]
        for i in range(len(info)):
            information = info[i]
            point = [float(information[0]), float(information[1]), float(information[2])]   
            pts.append(point)
        for j in range(0, len(pts)):
             colors.append(color)
        write_ply(pts,colors,None, './result.ply')
        #ply_to_pcd('./result.ply', output_path, img_id)
        #os.remove('./result.ply')
        path = './result.ply'
    #print("txt to pcd is done")
    return path
                    
def npy_to_pcd(input_path, output_path, img_id):
    data = np.load(path)
    output_path = output_path + '/' + str(img_id) + '.pcd'
    pts = []
    colors = []
    if len(data[0]) == 6:
        for i in range(len(data)):
            info = data[i]
            points = [info[0], info[1], info[2]]
            pts.append(points)
            color = [info[3], info[4], info[5]]
            colors.append(color)
    else:
         for i in range(len(data)):
            info = data[i]
            points = [info[0], info[1], info[2]]
            pts.append(points)
          #pts=pts   
         for j in range(0, len(pts)):
            color = [255, 255, 0]
            colors.append(color)
    write_ply(pts,colors,None, './result.ply')
    #ply_to_pcd('./result.ply', output_path, img_id)
    #os.remove('./result.ply')
    #print("npy_to_pcd is done")
    path = './result.ply'
    return path
                     
def read_off(file):
    if 'OFF' != file.readline().strip():
        raise('Not a vaild OFF header')
    n_verts, n_faces, _ = tuple([int(s) for s in file.readline().strip().split(' ')])
    verts = [[float(s) for s in file.readline().strip().split(' ')] for i_vert in range(n_verts)]
    faces = [[int(s) for s in file.readline().strip().split(' ')][1:] for i_face in range(n_faces)]
    
    return verts, faces

def off_to_pcd(input_dir, save_file_path, img_id):
    output_path = output_path + '/' + str(img_id) + '.pcd'
    with open(intput_dir, 'r') as f:
        verts, faces = read_off(f)
        
    if len(verts[0])==3:
        point = torch.tensor(verts)
        points = point[:,:3]
        
        colors=[]
        for i in range(0, points.shape[0]):
            color = [255,212,0]
            colors.append(color)
        write_ply(points, colors, None, './t.ply')
        path = './t.ply'
    else:
         infor = torch.tensor(verts)
         points = infor[:,:3]
         colors = infor[:,3:]
         write_ply(points, colors, None, './t.ply')
         path = './t.ply'
    #pcd = o3d.io.read_point_cloud('./t.ply')
    #o3d.io.write_point_cloud(output_path, pcd)
    #os.remove('./t.ply')
    
    #print("off_to_pcd is done")
    return path

import open3d as o3d
import os
import struct
import numpy as np
import shutil


#convert .bin to .pcd
def make_dataset_folder(directory):
    
    items = os.listdir(directory)
    items = [(os.path.join(directory, f)) for f in items]
    items = sorted(items)

    #print(f'Found {len(items)} folder imgs')

    return items

def bin_to_pcd(filepath):
    print(filepath)
    out_path = filepath[:-4] + '.pcd'
    print(out_path)
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
    

#ataroot = './data/mini_train/velodyne'

#aths = make_dataset_folder(dataroot)

#or path in paths:
#   bin_to_pcd(path)
    
#aths = make_dataset_folder(dataroot)

#or path in paths:
#   exten = paths.split('/')[-1].split('.')[-1]
#   if exten == 'bin':
#       os.remove(path)
#convert filename for upload
velodyne_path = make_dataset_folder('./data/mini_train/velodyne')
calib_path = make_dataset_folder('./data/mini_train/calib')
img_path = make_dataset_folder('./data/mini_train/image')

new_calib = './data/test_data/calib'
new_img = './data/test_data/image'
new_velo = './data/test_data/velodyne'

for folder in [new_calib, new_img, new_velo]:
    if not os.path.isdir(folder):
        os.makedirs(folder)

for i in range(len(velodyne_path)):
    new_calib_sub = new_calib + '/' + str(i)
    new_img_sub = new_img + '/' + str(i)
    
    for folder in [new_calib_sub, new_img_sub]:
        if not os.path.isdir(folder):
            os.makedirs(folder)


#file name must to be .lower --> [back, front, back_left, back_right, front_left, front_right]        
for velo in velodyne_path:
    name = velo.split('/')[-1].split('.')[-2]
    for i in range(len(img_path)):
        old = velo
        new = new_velo + '/' + str(i) + '.pcd'
        shutil.copy(old, new)
        calib_ = calib_path[i]
        print('check', calib_)
        img_ = img_path[i]
        print('why?',img_)
        
        calib = make_dataset_folder(calib_)
        img = make_dataset_folder(img_)
        print('ttt', calib)
        print('!!!!', img)
        
        for j in range(len(calib)):
            ca_path = calib[j]
            img_path = img[j]
            ca_name = ca_path.split('/')[-1].split('.')[-2]
            im_name = img_path.split('/')[-1].split('.')[-2]
            
            view_type = ca_path.split('/')[-2]
            
            if ca_name == name:
                new_calib = new_calib + '/' + str(i) + '/' + str(view_type) + '.txt'
                shutil.copy(ca_path, new_calib)
            if im_name == name:
                new_image = new_img + '/' + str(i) + '/' + str(view_type) + '.png'
                shutil.copy(img_path, new_image)   
        

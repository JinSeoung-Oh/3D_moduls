import numpy as np
from pyquaternion import Quaternion
from plyfile import PlyData

#only lidar data

def rotate(points, rot_matrix):
    points[:3, :] = np.dot(rot_matrix, points[:3, :])
   
    return points

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

lidar_pts = np.fromfile('./nuscenes/samples/LIDAR_TOP/n015-2018-11-21-19-58-31+0800__LIDAR_TOP__1542801733448313.pcd.bin', dtype=np.float32)

points = lidar_pts.reshape((-1,5))[:,:3]
space_rotate = Quaternion(axis=(0,0,1), angle = np.pi/2)
space_rotate_inv = space_rotate.inverse

points = rotate(points, space_rotate_inv.rotation_matrix)

colors = []
for i in range(len(points)):
    color = [0,255,255]  #random color
    colors.append(color)

write_ply(points, colors, None, './test_convert.ply')
convert_ply_to_bin('./test_convert.ply', '/test_convert', 0') 

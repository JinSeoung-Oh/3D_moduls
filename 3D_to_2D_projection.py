import math
import numpy as np
import numpy as np
import math
import argparse

#object by object
def parse_args():
    parser = argparse.ArgumentParser(description='3D_to_2D_projection')
    parser.add_argument('--calib_path', type=str, help='calib file path'')
    parser.add_argument('--img_path', type=str, help='image file path')
    parser.add_argument('--center', type=float, help='enter 3D center point')
    parser.add_argument('--dim', type= float, help = 'enter 3D dimention of object')
    parser.add_argument('--yaw', type=float, help = 'enter z_rotate')

    args = parser.parse_args()
    return args

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
    pts_2d = []
    location, dim, ry = label[0:3], label[3:6], label[6]
    corners_3d = compute_box_3d(dim, location, ry)
    corners_2d = project_to_image(corners_3d, calib.P2)
    corners_2d = corners_2d.tolist()
    corner = [corners_2d]
    pts_2d.append(corner)
                        
    return pts_2d

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


def roty(angle):
    # Rotation about the y-axis.
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]])


def compute_box_3d(dim, location, ry):
    R = roty(ry)
    h, w, l = dim
    cx,cy,cz = location
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2 ,-h / 2 , -h /2]
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
   
    corners_3d[0, :] = corners_3d[0, :] + cx
    corners_3d[1, :] = corners_3d[1, :] + cy
    corners_3d[2, :] = corners_3d[2, :] + cz

    return np.transpose(corners_3d,(1, 0))

                        
def lidar_to_camera_box(boxes,V2C=None, R0=None, P2=None):
    ret = []
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
        #p = np.matmul(R0, p)
    else:
        p = np.matmul(V2C, p)
        #p = np.matmul(R0, p)
    p = p[0:3]
   
    return tuple(p)

def project_to_image(pts_3d, P):
    # pts_3d: n x 3
    # P: 3 x 4
    # return: n x 2
    fx = P[0,0]
    fy = P[1,1]
    cx = P[0,2]
    cy = P[1,2]

    points = []
    for pts in pts_3d:
        x = pts[0] / pts[2]
        y = pts[1] / pts[2]

        u = x*fx + cx
        v = y*fy + cy

        points.append([int(u), int(v)])

    pts_2d = np.array(points)

    return pts_2d.astype(np.int64)
   
def main():
    args = parse_args()
    center_coor = args.center
    dim_ = args.dim
    yaw = args.yaw
                        
    center_coor = x,y,z
    dim_ = h,w,l
    
    box = [x,y,z,h,w,l,yaw]
    #get calib_information from calib_path
    #calib file style is KITTI style                        
    with open(args.calib) as f:
        lines = f.readlines()
    matrix = lines[0].strip().split(' ')[1:]
    P = np.array(matrix, dtype=np.float32)
    matrix = lines[1].strip().split(' ')[1:]
    R0 = np.array(matrix, dtype=np.float32)
    matrix = lines[2].strip().split(' ')[1:]
    Tr_velo_to_cam = np.array(matrix, dtype=np.float32)
                  
    info = lidar_to_camera_box(box, Tr_velo_to_cam, R0, P)
    informa = get_2d_pts(info, calib)
    information = informa[0][0]

    return information
                        
if __name__ == '__main__':
  main()

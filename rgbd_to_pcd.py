import cv2
import numpy as np
import open3d as o3d

def rgbd_to_pcd(depth_fx, depth_fy, depth_cx, depth_cy, rgb_fx, rgb_fy, rgb_cv, rgb_cy, rotation_matrix, transformation_matrix, out_path, type):
  depth_image = cv2.imread('depth_image path')
  rgb_image = cv2.imread('rgb_image path')
  out_path = f"{out_path[-1]}.{type}"
  R = -np.array(rotation_matrix)
  T = np.array(transformation_matrx)
  
  pcd = []
  color = []
  h, w, c = depth_image.shape
  
  for i in range(h):
    for j in range(w):
      #get points from depth image
      z = depth_image[i][j]
      x = (j-depth_cx) * z / depth_fx
      y = (i-depth_cy) * z / depth_fy
      pcd.append([x,y,z])
      
      #get color value from rgb image
      [x_rgb, y_rgb, z_rgb] = np.linalg.inv(R).dot([x,y,z])-np.linalg.inv(R).dot(T)
      
      rgb_j = int((x_rgb * rgb_fx) / z_rgb + rgb_cx + w/2)
      rgb_i = int((y_rgb * rgb_fy) / z_rgb + rgb_cy)
      
      if 0 <= rgb_j < w and 0<= rgb_i < h:
        color.append(rgb_image[i_rgb][j_rgb] / 255)
      else:
           continue
    
  point_cloud_data = o3d.geometry.PointCloud()
  point_cloud_data.points = o3d.utility.Vector3dVector(pcd)
  point_cloud_data.colors = o3d.utility.Vector3dVector(colors)
  pcd_data = np.array(point_cloud_data)
  
  o3d.io.write_point_cloud(out_path, pcd_data)

                         

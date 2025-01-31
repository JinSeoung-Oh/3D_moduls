import numpy as np
import torch


class Calibration(object):
    ''' Calibration matrices and utils
        3d XYZ in <label>.txt are in rect camera coord.
        2d box xy are in image2 coord
        Points in <lidar>.bin are in Velodyne coord.
        y_image2 = P^2_rect * x_rect
        y_image2 = P^2_rect * R0_rect * Tr_velo_to_cam * x_velo
        x_ref = Tr_velo_to_cam * x_velo
        x_rect = R0_rect * x_ref
        P^2_rect = [f^2_u,  0,      c^2_u,  -f^2_u b^2_x;
                    0,      f^2_v,  c^2_v,  -f^2_v b^2_y;
                    0,      0,      1,      0]
                 = K * [1|t]
        image2 coord:
         ----> x-axis (u)
        |
        |
        v y-axis (v)
        velodyne coord:
        front x, left y, up z
        rect/ref camera coord:
        right x, down y, front z
        Ref (KITTI paper): http://www.cvlibs.net/publications/Geiger2013IJRR.pdf
        TODO(rqi): do matrix multiplication only once for each projection.
    '''

    def __init__(self, calib_filepath):
        calibs = self.read_calib_file(calib_filepath)
        # Projection matrix from rect camera coord to image2 coord
        self.P2 = calibs['P2']
        self.tag = calibs['tag']
        
        # Camera intrinsics and extrinsics
        if self.tag == 'reference camera' :
            self.P2 = np.reshape(self.P2, [3,3])
            self.c_u = self.P2[0,2]
            self.c_v = self.P2[1,2]
            self.f_u = self.P2[0,0]
            self.f_v = self.P2[1,1]
        else:
            self.P2 = np.reshape(self.P2, [3,4])
            self.c_u = self.P2[0,2]
            self.c_v = self.P2[1,2]
            self.f_u = self.P2[0,0]
            self.f_v = self.P2[1,1]
            self.b_x = self.P2[0,3] / (-self.f_u)
            self.b_y = self.P2[1,3] / (-self.f_v)

        # Rigid transform from Velodyne coord to reference camera coord
        self.V2C = calibs['Tr_velo2cam']
        self.V2C = np.reshape(self.V2C, [3, 4])
        
        # Rotation from reference camera coord to rect camera coord
        self.R0 = calibs['R_rect']
        self.R0 = np.reshape(self.R0, [3, 3])

    def read_calib_file(self, filepath):
        with open(filepath) as f:
            lines = f.readlines()

        obj = lines[2].strip().split(' ')[1:]
        P2,tag = self.determin_calib(obj)
        #obj = lines[3].strip().split(' ')[1:]
        #P3 = np.array(obj, dtype=np.float32)
        obj = lines[4].strip().split(' ')[1:]
        R0 = np.array(obj, dtype=np.float32)
        obj = lines[5].strip().split(' ')[1:]
        Tr_velo_to_cam = np.array(obj, dtype=np.float32)

        return {'P2': P2,
                'tag': tag,
                #'P3': P3.reshape(3, 4),
                'R_rect': R0.reshape(3, 3),
                'Tr_velo2cam': Tr_velo_to_cam.reshape(3, 4)}

    def cart2hom(self, pts_3d):
        """
        :param pts: (N, 3 or 2)
        :return pts_hom: (N, 4 or 3)
        """
        pts_hom = np.hstack((pts_3d, np.ones((pts_3d.shape[0], 1), dtype=np.float32)))
        return pts_hom
        
    def determin_calib(self, matrix):
        if len(matrix) == 9:
            matrix = np.array(matrix, dtype=np.float32)
            matrix = matrix.reshape(3,3)
            tag = 'prime camera'
        else:
            if matrix[-1] == 0 and matrix[-2] == 0 and matrix[-3] == 0:
               matrix = np.array(matrix, dtype=np.float32)
               matrix = matrix.reshape(3,3)
               tag = 'prime camera'
            else:
                 matrix = np.array(matrix, dtype=np.float32)
                 matrix = matrix.reshape(3,4)
                 tag = 'not prime  camera'
                
        return matrix, tag
        
          

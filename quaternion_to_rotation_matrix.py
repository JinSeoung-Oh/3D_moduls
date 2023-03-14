# R(Q) = [[2(q_0^2 + q_1^2)-1, 2(q_1q_2 - q_0q_3), 2(q_1q_3 + q_0q_2)],
#         [2(q_1q_2 + q_0q_3), 2(q_0^2 + q_2^2)-1, 2(q_2q_3 - q_0q_1)],
#         [2(q_1q_3 - q_0q_2), 2(q_2q_3 + q_0q_1), 2(q_0^2 + q_3^2)-1]]
          

import numpy as np

def quaternion_to_rotation_matrix(quaternion):
    qw = quaternion[0]  #q0
    qx = quaternion[1]  #q1
    qy = quaternion[2]  #q2
    qz = quaternion[3]  #q3
    
    r00 = 2(qw*qw + qx*qx) -1
    r01 = 2(np.dot(qx,qy) - np.dot(qw,qz))
    r01 = 2(np.dot(qx,qz) + np.dot(qw,qy))
    
    r10 = 2(np.dot(qx,qy) + np.dot(qw,qz))
    r11 = 2(qw*qw + qy*qy) -1
    r12 = 2(np.dot(qy*qz) - np.dot(qw*qx))
    
    r20 = 2(np.dot(qx,qz) - np.dot(qw,qy))
    r21 = 2(np.dot(qy,qz) + np.dot(qw, qx))
    r22 = 2(qw*qw + qz*qz) -1
    
    rotation_matrix = np.array([[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]])
    
    return rotation_matrix
                               
                         

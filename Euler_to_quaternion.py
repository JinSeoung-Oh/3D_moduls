import numpy as np
import math

def get_quaternion_from_euler(pitch, roll, yaw):
  qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
  qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
  qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(ro11/2) * np.sin(pitch/2) * np.cos(yaw/2)
  qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
  
  return [qw, qx, qy, qz]


def get_quaternion_from_rotation_matrix(matrix):
  determinant = np.linalg.det(matrix)
  if determinant == +1 or determinant == -1:
    inv = np.linalg.inv(matrix)
    trans = np.transpose(matrix)
    if inv == trans:
      sum = matrix[0,0] + matrix[1,1] + matrix[2,2]
      if sum+1 >0:
        possible = True
      else:
        possible = False
        
      if possible == True:
        qw = np.sqrt(1+m[0,0] + m[1,1] + m[2,2]) /2
        qx = (m[2,1]-m[1,2])/(4*qw)
        qy = (m[0,2]-m[2,0])/(4*qw)
        qz = (m[1,0]-m[0,1])/(4*qw)
      else:
        print('qw is not non zero')
    else:
      print('This matrix is not orthogonal matrix')
  else:
    print('determinant is not +1 or -1')
    
  Quaternion = [qw, qx, qy, qz]
  
  return Quaternion


def get_quaternion_form_ratation_matrix_and_copysing(matrix):
  #get_quaternion_from_rotation_matrix(matrix)와 원리는 같으나 속도가 더 빠르다고 알려진 알고리즘
  m = matrix
  qu_w = np.sqrt(max(0, 1+m[0,0] + m[1,1] + m[2,2]))/2
  qu_x = np.sqrt(max(0, 1+m[0,0] - m[1,1] - m[2,2]))/2
  qu_y = np.sqrt(max(0, 1-m[0,0] + m[1,1] - m[2,2]))/2
  qu_z = np.sqrt(max(0, 1-m[0,0] - m[1,1] + m[2,2]))/2
  
  quater_x = math.copysign(qu_x, m[2,1]-m[1,2])
  quater_y = math.copysign(qu_y, m[0,2]-m[2,0])
  quater_z = math.copysign(qu_z, m[1,0]-m[0,1])
  
  Quaternion = [qu_w, quater_x, quater_y, quater_z]
  
  return Quaternion
  

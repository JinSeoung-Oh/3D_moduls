import numpy as np
import argparse

def parser():
    parser = argparse.ArgumentParser(description='parser for getting quaternion')
    parser.add_argument('--yaw', type=float, default=None, help='enter yaw value')
    parser.add_argument('--pitch', type=float, default=None, help='enter pitch value')
    parser.add_argument('--roll', type=float, default=None, help = 'enter roll value')
    parser.add_argument('--matrixdata', action='store', type=float, default=None, nargs='+', help='enter the matrix value. order is [0,0], [0,1], [0,2], ... ,[0,m]... [n,0],[n,1], ... ,[n,m] --> ex) 1 2 3 4 5 6 .. m')
    parser.add_argument('--nrows', action='store', type=int, default=None, help = 'number of row')
    args = parser.parse_args()
    
    return args

def get_marix(args):
    m = np.array(args.matrixdata).reshape((args.nrows, len(args.matrixdata)//args.nrows))
    
    return m

def get_quaternion_from_ratated_angle(pitch, roll, yaw):
    qx = np.sin(roll/2)*np.cos(pitch/2)*np.cos(yaw/2) - np.cos(roll/2)*np.sin(pitch/2)*np.sin(yaw/2)
    qy = np.cos(roll/2)*np.sin(pitch/2)*np.cos(yaw/2) + np.sin(roll/2)*np.cos(pitch/2)*np.sin(yaw/2)
    qz = np.cos(roll/2)*np.cos(pitch/2)*np.sin(yaw/2) - np.sin(roll/2)*np.sin(pitch/2)*np.cos(yaw/2)
    qw = np.cos(roll/2)*np.cos(pitch/2)*np.cos(yaw/2) + np.sin(roll/2)*np.sin(pitch/2)*np.sin(yaw/2)
    
    return [qw, qx, qy, qz]
                                          

def deter_3(matrix):
    a = np.array([matrix[0,0], matrix[0,1], matrix[0,2]], [matrix[1,0], matrix[1,1], matrix[1,2]], [matrix[2,0], matrix[2,1], matrix[2,2]])
    d = np.linalg.det(a)
    
    return d
    
def inv_3d(matrix):
    a = np.array([matrix[0,0], matrix[0,1], matrix[0,2]], [matrix[1,0], matrix[1,1], matrix[1,2]], [matrix[2,0], matrix[2,1],matrix[2,2]])
    inv_array = np.linalg.inv(a)
    
    return inv_array

def possible(marix):
    sum = marix[0,0] + matrix[1,1] + matrix[2,2]
    if sum+1 > 0:
        possi = 'True'
    else:
        possi = 'False'
       
    return possi
                                                            
def get_quaternion(matrix):
    qu_w = np.sqrt(max(0, 1+m[0,0]+m[1,1]+m[2,2]))/2
    qu_x = np.sqrt(max(0, 1+m[0,0]-m[1,1]+m[2,2]))/2
    qu_y = np.sqrt(max(0, 1-m[0,0]+m[1,1]-m[2,2]))/2
    qu_z = np.sqrt(max(0, 1-m[0,0]-m[1,1]+m[2,2]))/2
    
    quaternion_x = math.copysign(qu_x, m[2,1]-m[1,2])
    quaternion_y = math.copysign(qu_y, m[0,2]-m[2,0])
    quaternion_z = math.copysign(qu_z, m[1,0]-m[0,1])
    
    quaternion = [qu_w, quaternion_x, quaternion_y, quaternion_z]
    
    return quaternion
       
def main():
    args = parse_arg()
    yaw = args.yaw
    roll = args.roll
    pitch = args.pirch
    
    matrixdata = args.matrixdata
    if matrixdata is not None:
        matrix = get_matrix(args)
        determinant = deter_3(matrix)
        if determinant = +1 or determinant = -1:
            inv = inv_3d(matrix)
            a = np.array([matrix[0,0], matrix[0,1], matrix[0,2]], [matrix[1,0], matrix[1,1], matrix[1,2]], [matrix[2,0], matrix[2,1],matrix[2,2]])
            trans = np.transpose(a)
            if inv == trans:
                possibility = possible(marix)
                if possibility == True:
                    qw = np.sqrt(1+m[0,0] + m[1,1] + m[2,2]) / 2
                    qx = (m[2,1]-m[1,2])/(4*qw)
                    qy = (m[0,2]-m[2,0])/(4*qw)
                    qz = (m[1,0] -m[0,1])/(4*qw)
                else:
                    #Quaternions = get_quaternion(matrix)
                    yaw = np.arctan2(matrix[2,1], matrix[1,1])
                    pitch = np.arctan2(-matrix[3,1], np.sqrt(matrix[3,2]**2 + matrix[3,3]**2))
                    roll = np.arctan2(matrix[3,2], matrix[3,3])
                    Quaternions = get_quaternion_from_ratated_angle(pitch, roll, yaw)
                                                           
            else:
                yaw = np.arctan2(matrix[2,1], matrix[1,1])
                pitch = np.arctan2(-matrix[3,1], np.sqrt(matrix[3,2]**2 + matrix[3,3]**2))
                roll = np.arctan2(matrix[3,2], matrix[3,3])
                Quaternions = get_quaternion_from_ratated_angle(pitch, roll, yaw)
        else:
            yaw = np.arctan2(matrix[2,1], matrix[1,1])
            pitch = np.arctan2(-matrix[3,1], np.sqrt(matrix[3,2]**2 + matrix[3,3]**2))
            roll = np.arctan2(matrix[3,2], matrix[3,3])
            Quaternions = get_quaternion_from_ratated_angle(pitch, roll, yaw)
        
        Quaternions = [qw, qx, qy, qz]
        
    if yaw is not None:
        Quaternions = get_quaternion_from_ratated_angle(pitch, roll, yaw)
    
    return print('quaternions is:' Quaternions)
    
if __name__ == '__main__':
    main()

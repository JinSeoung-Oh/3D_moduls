import math
import numpy as np
import torch
import argparse
import json

def parser():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--out_path', type=str, default=None, help='enter your save path')
    parser.add_argument('--camera', type=int, default=None, help="enter the camera position like (304,304")
    parser.add_argument('--center',  type=str, help='Enter the object center') 
    parser.add_argument('--front', type=int, default=None, help="enter the object front like (331, 459")
    args = parser.parse_args()
    
    return args


def getAngle2P(p1,p2, direction='CW'):
    ang1 = np.arctan2(*p1[::-1])
    ang2 = np.arctan2(*p2[::-1])
    res = np.rad2deg((ang1-ang2)%(2*np.pi))
    if direction=="CCW":
        res = (360-res)%360
    return res


def main():
    args = parser()
    save_path = args.out_path
    cemera_position = args.camera
    object_center = args.center
    object_front = args.front
    
    #camera_position = (304,304)
    #object_center = (330,475)
    #object_front = (331,459)
    
    adjust_origin_x = 0-camera_position[0]
    adjust_origin_y = 0-camera_position[1]

    if adjust_origin_x > 0:
        adjust_object_center_x = object_center[0] - adjust_origin_x
        adjust_front_x = object_front[0] - adjust_origin_x
    else:
        adjust_object_center_x = object_center[0] + adjust_origin_x
        adjust_front_x = object_front[0] + adjust_origin_x

    if adjust_origin_y > 0:
        adjust_object_center_y = object_center[1] - adjust_origin_y
        adjust_front_y = object_front[1] - adjust_origin_y
    else:
        adjust_object_center_y = object_center[1] + adjust_origin_y
        adjust_front_y = object_front[1] + adjust_origin_y

    adjust_center_point = (adjust_object_center_x, adjust_object_center_y)
    adjust_front_point = (adjust_front_x, adjust_front_y)

    alpa_control = (0, adjust_center_point[1])

    move_x = 0-adjust_center_point[0]
    move_y = 0-adjust_center_point[1]

    move_center_x = adjust_center_point[0] + move_x
    move_front_x = adjust_front_point[0] + move_x
    move_center_y = adjust_center_point[1] + move_y
    move_front_y = adjust_front_point[1] + move_y

    move_center = [move_center_x, move_center_y]
    move_front = [move_front_x, move_front_y]
    control_point = [0, move_front[1]]

    if adjust_center_point[0] > 0 :
        alpa = getAngle2P(adjust_center_point,alpa_control)
        right_system_obj_angle = math.radians(alpa)
        left_system_obj_angle = -math.radians(alpa) - (math.pi/2)
        rot_z = getAngle2P(move_front,control_point)
        right_system_rot_z = math.radians(rot_z)
        rot_y = -math.radians(right_system_rot_z) - (math.pi/2)
    #print('????')
    else:
        alpa = getAngle2P(adjust_center_point,alpa_control,"CCW")
        right_system_obj_angle = math.radians(alpa)
        left_system_obj_angle = -math.radians(alpa) - (math.pi/2)
        rot_z = getAngle2P(move_front,control_point, "CCW")
        right_system_rot_z = math.radians(rot_z)
        rot_y = -math.radians(right_system_rot_z) - (math.pi/2)
    #print('!!!!')
    
    if adjust_center_point[0] > 0 and adjust_center_point[1]>0:
        left_system_obj_angle = left_system_obj_angle
        rot_y = rot_y
    elif adjust_center_point[0] < 0 and adjust_center_point[1]>0:
        left_system_obj_angle = -(left_system_obj_angle)
        rot_y = -(rot_y)
    elif adjust_center_point[0] <0 and adjust_center_point[1]<0:
        left_system_obj_angle = -(left_system_obj_angle)
        rot_y = -(rot_y)
    else:
        left_system_obj_angle = left_system_obj_angle
        rot_y = rot_y
    
    angle={}
    angle['left_objective_angle'] = left_system_obj_angle
    angle['right_objective_angle'] = right_system_obj_angle
    angle['rotated_y'] = rot_y
    angle['roteted_z'] = rot_z
    
    with open(save_path, 'w') as f:
        json.dump(angle, f)
        

if __name__ == "__main__":
    main()

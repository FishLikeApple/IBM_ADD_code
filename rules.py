from helpers_and_configurations import *

def rule1(input_coords, D1=D1, car_width=6):
  
    warning_coords = []
    for input_coord in input_coords:
        if input_coord['z']<D1 and np.aps(input_coord['x'])<car_width*1.1:
          warning_coords.append(input_coord)
          
    return warning_coords

def rule2(input_coords, D2=D2):
  
    warning_coords = []
    for input_coord in input_coords:
        if input_coord['z'] < D1:
          pitch = input_coord['pitch'] + np.pi/2
#          if np.abs(pitch)<np.pi/2 and input_coord[]:

from helpers_and_configurations import *

def rule1(input_coords, warning_coords=[], D=D1, judgment_width=3):
  
    for input_coord in input_coords:
        if input_coord not in warning_coords:
            if input_coord['z']<D and np.abs(input_coord['x'])<judgment_width:
                warning_coords.append(input_coord)
          
    return warning_coords

def rule2(input_coords, warning_coords=[], D=D2, pitch_threshold=0.1):
  
    for input_coord in input_coords:
        if input_coord not in warning_coords:
            if input_coord['z'] < D:
                if input_coord['x'] > 0:
                    pitch = input_coord['pitch'] - np.pi/2
                else:
                    pitch = input_coord['pitch'] + np.pi/2
                if np.abs(np.arctan(input_coord['y']/input_coord['x'])-pitch) < pitch_threshold:
                    warning_coords.append(input_coord)
          
    return warning_coords
  
def rule3(input_coords, warning_coords=[], D=D3, pitch_threshold=0.2):
  
    warning_coords = []
    for input_coord in input_coords:
        if input_coord['z'] < D:
            if input_coord['x'] > 0:
                pitch = -input_coord['pitch']
            else:
                pitch = input_coord['pitch']
            if pitch<np.pi+pitch_threshold and pitch>0-pitch_threshold:
                warning_coords.append(input_coord)
                    
    return warning_coords

def rule4(input_coords, warning_coords=[], D=D4, pitch_threshold=0.2):
    # actually the same as the above func
  
    warning_coords = []
    for input_coord in input_coords:
        if input_coord['z'] < D:
            if input_coord['x'] > 0:
                pitch = -input_coord['pitch']
            else:
                pitch = input_coord['pitch']
            if pitch<np.pi+pitch_threshold and pitch>0-pitch_threshold:
                warning_coords.append(input_coord)
                   
    return warning_coords

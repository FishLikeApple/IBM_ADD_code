from helpers import *

if not use_ID_thr:
    def is_close_enough(a, b, TD_thr=TD_thr, RD_thr=RD_thr):
        # judge if a and b are close enough.

        RD = RotationDistance(a, b)
        if RD > RD_thr:
            return False, None, None
        TD = TranslationDistance(a, b)
        if TD > TD_thr:
            return False, None, None
        return True, TD, RD
else:
    # use image coordinates
    def is_close_enough(a, b, thr_factor=ID_thr_factor):
        # judge if a and b are close enough.

        RD = RotationDistance(a, b)
        if RD > RD_thr:
            return False, None, None

        img_x_a, img_y_a = get_img_coords([a])
        img_x_b, img_y_b = get_img_coords([b])
        distance = sqrt((img_x_a-img_x_b)*(img_x_a-img_x_b)+(img_y_a-img_y_b)*(img_y_a-img_y_b))
        #print(distance)

        if distance > (5400.0/a['z'])*thr_factor:
            return False, None, None

        return True, distance, None
    
# the main function
def output_merging(A_coords, M_coords):
    
    new_M_coords = M_coords.copy()
    for A_coord in A_coords:
        candidate = {'candidate':None, 'TD':99999999}
        for j in range(len(M_coords)):
            M_coord = M_coords[j]
            enough, TD, RD = is_close_enough(A_coord, M_coord)
            #print(enough)
            if enough and M_coord not in paired_M_coords and candidate['TD']>TD:
                candidate['candidate'] = M_coord
                candidate['TD'] = TD
                candidate['index'] = j
        if candidate['candidate'] == None:
            new_M_coords.append(A_coord)
        else:
            M_coord = candidate['candidate']
            paired_M_coords.append(M_coord)
            if merging_plan == 'weighted-add':
                new_coord = {}
                for item in ['pitch','yaw','roll','x','y','z']:
                    total_score = A_coord['confidence'] + M_coord['confidence']
                    new_coord[item] = (A_coord[item]*(A_coord['confidence']/total_score))\
                                    + (M_coord[item]*(M_coord['confidence']/total_score))       
            else:
                if A_coord['confidence'] > M_coord['confidence']:
                    new_coord = A_coord
                else:
                    new_coord = M_coord
            new_coord['confidence'] = 1 - ((1-A_coord['confidence'])*(1-M_coord['confidence']))
            new_M_coords[candidate['index']] = new_coord
            
    return new_M_coords

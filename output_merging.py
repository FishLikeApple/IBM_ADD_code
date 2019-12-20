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

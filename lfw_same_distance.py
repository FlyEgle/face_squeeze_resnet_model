import numpy as np
from lfw_evalution import person_image_index, calcute_distance

if __name__ == '__main__':
    weights_path = '/Users/jmc/Desktop/facepaper/face_data/weights/best_softmax+0.01_centerloss.h5'
    path = '/Users/jmc/Desktop/facepaper/lwf_pair.txt'
    same_person, not_same_person = person_image_index(path)
    same_distance = calcute_distance(same_person,
                                     weight=weights_path,
                                     mode='cosine')

    np.save('/Users/jmc/Desktop/facepaper/face_data/weights/same_distance_dis_new_0_0_1_s_centerloss.npy', same_distance)

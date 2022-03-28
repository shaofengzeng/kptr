import cv2
import numpy as np
import solver.descriptor_evaluation as ev
from utils.plt import plot_imgs

def draw_matches(data):
    keypoints1 = [cv2.KeyPoint(p[1], p[0], 1) for p in data['keypoints1']]
    keypoints2 = [cv2.KeyPoint(p[1], p[0], 1) for p in data['keypoints2']]
    inliers = data['inliers'].astype(bool)
    matches = np.array(data['matches'])[inliers].tolist()
    img1 = cv2.merge([data['image1'], data['image1'], data['image1']]) * 255
    img2 = cv2.merge([data['image2'], data['image2'], data['image2']]) * 255
    return cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches,
                           None, matchColor=(0,255,0), singlePointColor=(0, 0, 255))


experiments = ['./data/descriptors/hpatches/kptr/']

## draw matching examples
num_images = 5
for e in experiments:
    orb = True if e[:3] == 'orb' else False
    outputs = ev.get_homography_matches(e, keep_k_points=1000, correctness_thresh=3, num_images=num_images, orb=orb)
    for ind, output in enumerate(outputs):
        img = draw_matches(output) / 255.
        img_w = (img*255).astype(np.int).astype(np.uint8)
        cv2.imwrite('{}.jpg'.format(ind),img_w)
        #plot_imgs([img], titles=[e], dpi=200)
print('Draw Done')


##Homography estimation correctness
for exp in experiments:
    orb = True if exp[:3] == 'orb' else False
    correctness = ev.homography_estimation(exp, keep_k_points=1000, correctness_thresh=5, orb=orb)
    print('> {}: {}'.format(exp, correctness))


# ##Check that the image is warped correctly
# num_images = 2
# for e in experiments:
#     orb = True if e[:3] == 'orb' else False
#     outputs = ev.get_homography_matches(e, keep_k_points=1000, correctness_thresh=3, num_images=num_images, orb=orb)
#     for output in outputs:
#         img1 = output['image1'] * 255
#         img2 = output['image2'] * 255
#         H = output['homography']
#         warped_img1 = cv2.warpPerspective(img1, H, (img2.shape[1], img2.shape[0]))
#         img1 = np.concatenate([img1, img1, img1], axis=2)
#         warped_img1 = np.stack([warped_img1, warped_img1, warped_img1], axis=2)
#         img2 = np.concatenate([img2, img2, img2], axis=2)
#         plot_imgs([img1 / 255., img2 / 255., warped_img1 / 255.], titles=['img1', 'img2', 'warped_img1'], dpi=200)
#
# ##Homography estimation correctness
# for exp in experiments:
#     orb = True if exp[:3] == 'orb' else False
#     correctness = ev.homography_estimation(exp, keep_k_points=1000, correctness_thresh=3, orb=orb)
#     print('> {}: {}'.format(exp, correctness))

#kptr_5.pth
#det_thresh:0.001, c_thresh:1
#hpatches-v:
#hpatches-i:
#hpatches:0.45

#det_thresh:0.001, c_thresh:3
#hpatches-v:0.712
#hpatches-i:0.947
#hpatches:0.829

#det_thresh:0.001, c_thresh:5
#hpatches-v:
#hpatches-i:
#hpatches:0.910

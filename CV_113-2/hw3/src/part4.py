import numpy as np
import cv2
import random
from tqdm import tqdm
from utils import solve_homography, warping

random.seed(999)

def panorama(imgs):
    """
    Image stitching with estimated homograpy between consecutive
    :param imgs: list of images to be stitched
    :return: stitched panorama
    """
    max_height = max([img.shape[0] for img in imgs])
    total_width = sum([img.shape[1] for img in imgs])

    # Create the overall canvas for the panorama
    canvas = np.zeros((max_height, total_width, imgs[0].shape[2]), dtype=np.uint8)
    canvas[:imgs[0].shape[0], :imgs[0].shape[1]] = imgs[0]
    cumulative_transform = np.eye(3)
    panorama_image = None

    current_offset = 0
    orb_detector = cv2.ORB_create()
    bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    # Process each consecutive image pair
    for idx in tqdm(range(len(imgs) - 1)):
        img_A = imgs[idx]
        img_B = imgs[idx + 1]
        current_offset += img_A.shape[1]

        # Step 1: Detect features and match descriptors
        kpts_A, desc_A = orb_detector.detectAndCompute(img_A, None)
        kpts_B, desc_B = orb_detector.detectAndCompute(img_B, None)
        raw_matches = bf_matcher.knnMatch(desc_A, desc_B, k=2)
        
        matched_A = []
        matched_B = []
        for m, n in raw_matches:
            if m.distance < 0.75 * n.distance:
                matched_A.append(kpts_A[m.queryIdx].pt)
                matched_B.append(kpts_B[m.trainIdx].pt)
                
        pts_A = np.array(matched_A)
        pts_B = np.array(matched_B)

        # Step 2: Use RANSAC to determine the best homography matrix
        num_trials = 5000
        error_thresh = 4
        best_inlier_count = 0
        best_H_candidate = np.eye(3)
        for trial in range(num_trials + 1):
            sample_A = np.zeros((4, 2))
            sample_B = np.zeros((4, 2)) 
            for j in range(4):
                rand_idx = random.randint(0, len(pts_A) - 1)
                sample_A[j] = pts_A[rand_idx]
                sample_B[j] = pts_B[rand_idx]
            H_candidate = solve_homography(sample_B, sample_A)

            one_row = np.ones((1, len(pts_A)))
            pts_B_homogeneous = np.concatenate((np.transpose(pts_B), one_row), axis=0)
            pts_A_homogeneous = np.concatenate((np.transpose(pts_A), one_row), axis=0)
            projected_pts = np.dot(H_candidate, pts_B_homogeneous)
            # Check for zero values in the scaling factor to avoid division by zero
            if np.any(projected_pts[-1, :] == 0):
                continue
            projected_pts = np.divide(projected_pts, projected_pts[-1, :])
            
            reproj_error = np.linalg.norm((projected_pts - pts_A_homogeneous)[:-1, :], ord=1, axis=0)
            inlier_count = sum(reproj_error < error_thresh)
            inlier_coords_A = pts_A[reproj_error < error_thresh]
            inlier_coords_B = pts_B[reproj_error < error_thresh]
            
            if inlier_count > best_inlier_count:
                best_inlier_count = inlier_count
                best_H_candidate = H_candidate

        # Step 3: Update the cumulative transformation with the best current homography
        cumulative_transform = cumulative_transform.dot(best_H_candidate)
        # Step 4: Warp the current image into the panorama canvas using backward warping
        out = warping(img_B, canvas, cumulative_transform, 0, img_B.shape[0], current_offset, current_offset + img_B.shape[1], direction='b')

    return out 

if __name__ == "__main__":
    # ================== Part 4: Panorama ========================
    # TODO: change the number of frames to be stitched
    FRAME_NUM = 3
    imgs = [cv2.imread('../resource/frame{:d}.jpg'.format(x)) for x in range(1, FRAME_NUM + 1)]
    output4 = panorama(imgs)
    cv2.imwrite('output4.png', output4)
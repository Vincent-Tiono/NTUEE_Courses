import numpy as np
import cv2
import argparse
import os
from JBF import Joint_bilateral_filter
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description='main function of joint bilateral filter')
    parser.add_argument('--image_path', default='./testdata/2.png', help='path to input image')
    parser.add_argument('--setting_path', default='./testdata/2_setting.txt', help='path to setting file')
    args = parser.parse_args()

    img = cv2.imread(args.image_path)
    img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    ### TODO ###
    # Read settings: create R, G, B lists and get sigma_s, sigma_r
    R, G, B = [], [], []
    with open(args.setting_path) as f:
        # Skip header line
        for line in f.readlines()[1:]:
            s = line.strip().split(',')
            if 'sigma_s' in line:
                sigma_s = int(s[1])
                sigma_r = float(s[3])
            else:
                R.append(float(s[0]))
                G.append(float(s[1]))
                B.append(float(s[2]))

    # Create guidance images (weighted combinations of the R, G, B channels, plus plain grayscale)
    h, w = img_gray.shape
    num_guidances = len(R) + 1
    guidance = np.zeros((h, w, num_guidances), dtype=img_gray.dtype)
    for i in range(len(R)):
        guidance[:, :, i] = R[i] * img_rgb[:, :, 0] + G[i] * img_rgb[:, :, 1] + B[i] * img_rgb[:, :, 2]
    guidance[:, :, -1] = img_gray

    # Create JBF instance and compute the ground truth bilateral filtering result
    JBF = Joint_bilateral_filter(sigma_s, sigma_r)
    bf_out = JBF.joint_bilateral_filter(img_rgb, img_rgb).astype(np.uint8)

    # Apply joint bilateral filter on each guidance image; stack output along a new last axis
    guidance_jbf_out = np.stack([
        JBF.joint_bilateral_filter(img_rgb, guidance[:, :, i]).astype(np.uint8)
        for i in range(num_guidances)
    ], axis=-1)  # Shape: [h, w, 3, num_guidances]

    # Compute L1 norm (cost) for each filtered image versus the ground truth (bf_out)
    L1_norm = [
        np.sum(np.abs(bf_out.astype('int32') - guidance_jbf_out[:, :, :, i].astype('int32')))
        for i in range(num_guidances)
    ]
    print("L1_norm = ", L1_norm)

    # Determine indices for the highest and lowest cost guidance
    max_index = L1_norm.index(max(L1_norm))
    min_index = L1_norm.index(min(L1_norm))

    # Plot the results
    plt.subplot(2, 2, 1)
    plt.title('highest')
    plt.imshow(guidance_jbf_out[:, :, :, max_index])
    plt.axis('off')
    
    plt.subplot(2, 2, 2)
    plt.title('highest' + str(max_index + 1))
    plt.imshow(guidance[:, :, max_index], cmap='gray')
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.title('lowest')
    plt.imshow(guidance_jbf_out[:, :, :, min_index])
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.title('lowest' + str(min_index + 1))
    plt.imshow(guidance[:, :, min_index], cmap='gray')
    plt.axis('off')

    plt.show()


if __name__ == '__main__':
    main()
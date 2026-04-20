# ============================================================================
# File: util.py
# Date: 2025-03-11
# Author: TA
# Description: Utility functions to process BoW features and KNN classifier.
# ============================================================================

import numpy as np
from PIL import Image
from tqdm import tqdm
from cyvlfeat.sift.dsift import dsift
from cyvlfeat.kmeans import kmeans
from scipy.spatial.distance import cdist
from time import time

CAT = ['Kitchen', 'Store', 'Bedroom', 'LivingRoom', 'Office',
       'Industrial', 'Suburb', 'InsideCity', 'TallBuilding', 'Street',
       'Highway', 'OpenCountry', 'Coast', 'Mountain', 'Forest']

CAT2ID = {v: k for k, v in enumerate(CAT)}

########################################
###### FEATURE UTILS              ######
###### use TINY_IMAGE as features ######
########################################

###### Step 1-a
def get_tiny_images(img_paths: str):
    '''
    Build tiny image features.
    - Args: : 
        - img_paths (N): list of string of image paths
    - Returns: :
        - tiny_img_feats (N, d): ndarray of resized and then vectorized
                                 tiny images
    NOTE:
        1. N is the total number of images
        2. if the images are resized to 16x16, d would be 256
    '''
    
    #################################################################
    # TODO:                                                         #
    # To build a tiny image feature, you can follow below steps:    #
    #    1. simply resize the original image to a very small        #
    #       square resolution, e.g. 16x16. You can either resize    #
    #       the images to square while ignoring their aspect ratio  #
    #       or you can first crop the center square portion out of  #
    #       each image.                                             #
    #    2. flatten and normalize the resized image.                #
    #################################################################

    tiny_img = []
    for path in img_paths:
        img = Image.open(path).convert('L')  # Convert to grayscale
        tiny_img2D = img.resize((16, 16), Image.Resampling.LANCZOS)  # LANCZOS is similar to INTER_AREA
        tiny_img1D = np.array(tiny_img2D).flatten()
        tiny_img.append(tiny_img1D)

    tiny_img_feats = np.matrix(tiny_img)

    #################################################################
    #                        END OF YOUR CODE                       #
    #################################################################

    return tiny_img_feats

#########################################
###### FEATURE UTILS               ######
###### use BAG_OF_SIFT as features ######
#########################################

###### Step 1-b-1
def build_vocabulary(
        img_paths: list, 
        vocab_size: int = 400
    ):
    '''
    Args:
        img_paths (N): list of string of image paths (training)
        vocab_size: number of clusters desired
    Returns:
        vocab (vocab_size, sift_d): ndarray of clusters centers of k-means
    '''
    
    sift_feat = []
    print("Starting vocabulary building...")
    start_time = time()
    
    for image_path in tqdm(img_paths):
        img = Image.open(image_path).convert('L')
        img = np.array(img, dtype=np.float32)
        _, desc = dsift(img, step=[5, 5], fast=True)
        if desc.shape[0] > 128:
            selected_indices = np.random.choice(desc.shape[0], 128, replace=False)
            desc = desc[selected_indices]
        sift_feat.append(desc)

    sift_feat = np.vstack(sift_feat)

    vocab = kmeans(sift_feat.astype(np.float32), num_centers=vocab_size)
    end_time = time()
    print(f"Vocabulary building completed in {end_time - start_time:.2f} seconds")
    
    return vocab

###### Step 1-b-2
def get_bags_of_sifts(
        img_paths: list,
        vocab: np.array
    ):
    '''
    Args:
        img_paths (N): list of string of image paths
        vocab (vocab_size, sift_d) : ndarray of clusters centers of k-means
    Returns:
        img_feats (N, d): ndarray of feature of images, each row represent
                          a feature of an image, which is a normalized histogram
                          of vocabularies (cluster centers) on this image
    '''

    
    img_feats = []
    vocab_size = vocab.shape[0]
    start_time = time()
    print("Construct bags of sifts...")
    for img_path in tqdm(img_paths):
        img = Image.open(img_path).convert('L')
        img = np.array(img, dtype=np.float32)
        _, desc = dsift(img, step=[5, 5], fast=True)

        if desc is not None:
            distances = cdist(desc, vocab)
            nearest_vocab_indices = np.argmin(distances, axis=1)
            histogram, _ = np.histogram(nearest_vocab_indices, bins=np.arange(vocab_size + 1))
            histogram = histogram / np.sum(histogram)
            img_feats.append(histogram)

    return np.array(img_feats)


################################################
###### CLASSIFIER UTILS                   ######
###### use NEAREST_NEIGHBOR as classifier ######
################################################

###### Step 2
def nearest_neighbor_classify(
        train_img_feats: np.array,
        train_labels: list,
        test_img_feats: list
    ):
    '''
    Args:
        train_img_feats (N, d): ndarray of feature of training images
        train_labels (N): list of string of ground truth category for each 
                          training image
        test_img_feats (M, d): ndarray of feature of testing images
    Returns:
        test_predicts (M): list of string of predict category for each 
                           testing image
    NOTE:
        1. d is the dimension of the feature representation, depending on using
           'tiny_image' or 'bag_of_sift'
        2. N is the total number of training images
        3. M is the total number of testing images
    '''
    k=1
    test_predicts = []

    distances = cdist(test_img_feats, train_img_feats, metric='minkowski', p=0.4)
    
    for i in range(distances.shape[0]):
        indices = np.argsort(distances[i])[:k]
        labels = [train_labels[idx] for idx in indices]
        predicted_label = max(set(labels), key=labels.count)
        test_predicts.append(predicted_label)
    return test_predicts
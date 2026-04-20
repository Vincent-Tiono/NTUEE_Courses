import numpy as np
import cv2

class Difference_of_Gaussian(object):
    def __init__(self, threshold):
        self.threshold = threshold
        self.sigma = 2**(1/4)
        self.num_octaves = 2
        self.num_DoG_images_per_octave = 4
        self.num_guassian_images_per_octave = self.num_DoG_images_per_octave + 1

    def get_keypoints(self, image):
        ### TODO ####
        # Step 1: Filter images with different sigma values (5 images per octave, 2 octave in total)
        # - Function: cv2.GaussianBlur (kernel = (0, 0), sigma = self.sigma**___)
        num_gauss = self.num_guassian_images_per_octave
        # First octave: original image + blurred images with increasing sigma
        first_octave = [image] + [cv2.GaussianBlur(image, (0, 0), self.sigma**i) for i in range(1, num_gauss)]
        
        # Down-sample the last image from first octave
        ds_image = cv2.resize(first_octave[-1],
                            (image.shape[1] // 2, image.shape[0] // 2),
                            interpolation=cv2.INTER_NEAREST)
        
        # Second octave: down-sampled image + blurred versions
        second_octave = [ds_image] + [cv2.GaussianBlur(ds_image, (0, 0), self.sigma**i) for i in range(1, num_gauss)]
        
        # Combine octaves
        gaussian_images = [first_octave, second_octave]
        
        # Step 2: Subtract 2 neighbor images to get DoG images (4 images per octave, 2 octave in total)
        # - Function: cv2.subtract(second_image, first_image)
        dog_images = []
        # iterate through each octave for subtraction
        for i, GI in enumerate(gaussian_images):
            dog_image = []
            for j in range(self.num_DoG_images_per_octave):
                dog = cv2.subtract(GI[j], GI[j+1])
                dog_image.append(dog)
                # save DoG images to disk
                dog_min, dog_max = np.min(dog), np.max(dog)
                norm = (dog - dog_min) * 255 / (dog_max - dog_min)
                cv2.imwrite(f'testdata/DoG{i+1}-{j+1}.png', norm)
            dog_images.append(dog_image)

        # Step 3: Thresholding the value and Find local extremum (local maximun and local minimum)
        #         Keep local extremum as a keypoint
        
        all_keypoints = []
        for i in range(self.num_octaves):
            dogs = np.array(dog_images[i])
            cube = np.stack([
                np.roll(dogs, shift=(x, y, z), axis=(2, 1, 0))
                for z in range(-1, 2)
                for y in range(-1, 2)
                for x in range(-1, 2)
            ], axis=0)
            mask = (np.abs(dogs) >= self.threshold) & (
                (dogs == np.min(cube, axis=0)) | (dogs == np.max(cube, axis=0))
            )
            for j in range(1, self.num_DoG_images_per_octave-1):
                m = mask[j]
                # Use np.nonzero to directly get the indices where mask is True
                ys, xs = np.nonzero(m)
                kp = np.stack([ys, xs], axis=1)
                if i:
                    kp *= 2
                all_keypoints.append(kp)
        keypoints = np.vstack(all_keypoints) if all_keypoints else np.empty((0,2), dtype='int64')
                    
        # Step 4: Delete duplicate keypoints
        # - Function: np.unique
        keypoints = np.unique(np.array(keypoints), axis = 0)

        # sort 2d-point by y, then by x
        keypoints = keypoints[np.lexsort((keypoints[:,1],keypoints[:,0]))] 
        return keypoints

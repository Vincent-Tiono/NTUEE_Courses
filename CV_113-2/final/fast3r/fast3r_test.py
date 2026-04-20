import torch
from fast3r.dust3r.utils.image import load_images
from fast3r.dust3r.inference_multiview import inference
from fast3r.models.fast3r import Fast3R
from fast3r.models.multiview_dust3r_module import MultiViewDUSt3RLitModule
import argparse
import os

import glob

def main(input_dir, output_dir):
    # --- Setup ---
    # Load the model from Hugging Face
    model = Fast3R.from_pretrained("jedyang97/Fast3R_ViT_Large_512")  # If you have networking issues, try pre-download the HF checkpoint dir and change the path here to a local directory
    print("downloaded model\n")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = model.to(device)

    # Create a lightweight lightning module wrapper for the model.
    # This provides functions to estimate camera poses, evaluate 3D reconstruction, etc.
    lit_module = MultiViewDUSt3RLitModule.load_for_inference(model)

    # Set model to evaluation mode
    model.eval()
    lit_module.eval()

    # --- Load Images ---
    filelist = sorted(glob.glob(os.path.join(input_dir, "*.color.png")))

    images = load_images(filelist, size=244, verbose=True)

    # --- Run Inference ---
    # The inference function returns a dictionary with predictions and view information.
    output_dict, profiling_info = inference(
        images,
        model,
        device,
        dtype=torch.float32,  # or use torch.bfloat16 if supported
        verbose=True,
        profiling=True,
    )

    # --- Estimate Camera Poses ---
    # This step estimates the camera-to-world (c2w) poses for each view using PnP.
    poses_c2w_batch, estimated_focals = MultiViewDUSt3RLitModule.estimate_camera_poses(
        output_dict['preds'],
        niter_PnP=100,
        focal_length_estimation_method='first_view_from_global_head'
    )
    # poses_c2w_batch is a list; the first element contains the estimated poses for each view.
    camera_poses = poses_c2w_batch[0]

    # Print camera poses for all views.
    for view_idx, pose in enumerate(camera_poses):
        # f = open(f"./poses/vincent/1/frame-{view_idx:06d}.pose.txt", "w")
        
        f = open(f"{output_dir}/frame-{view_idx:06d}.pose.txt", "w")
        for i in range(4):
            for j in range(4):
                f.write(f"{pose[i][j]} ")
            f.write("\n")
        f.close()

    # --- Extract 3D Point Clouds for Each View ---
    # Each element in output_dict['preds'] corresponds to a view's point map.
    for view_idx, pred in enumerate(output_dict['preds']):
        point_cloud = pred['pts3d_in_other_view'].cpu().numpy()
        print(f"Point Cloud Shape for view {view_idx}: {point_cloud.shape}")  # shape: (1, 368, 512, 3), i.e., (1, Height, Width, XYZ)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Fast3R pose estimation.")
    parser.add_argument('--image_dir', type=str, required=True, help='Directory with input RGB images (*.color.png)')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save pose output')
    
    args = parser.parse_args()
    main(args.image_dir, args.output_dir)

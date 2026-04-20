import numpy as np
import os
import glob
import argparse

def main(poses_dir, aligned_dir):
    # --- clear folder ---
    pattern = os.path.join(aligned_dir, "frame-*.pose.txt")
    matching_files = glob.glob(pattern)

    for file_path in matching_files:
        filename = os.path.basename(file_path)
        if len(filename) == len("frame-000000.pose.txt"):  # 檢查是否剛好對應
            os.remove(file_path)
            print(f"Deleted: {file_path}")

    rel_pose_paths = sorted(glob.glob(f"{poses_dir}/frame-*.pose.txt"))
    rel_poses = [np.loadtxt(p) for p in rel_pose_paths]

    # --- Load ---
    gt_pose = np.loadtxt(f"{poses_dir}/ground_truth_0.pose.txt")
    rel_pose_paths = sorted(glob.glob(f"{poses_dir}/frame-*.pose.txt"))
    rel_poses = [np.loadtxt(p) for p in rel_pose_paths]

    # --- Compute alignment matrix ---
    T_gt = gt_pose
    T_pred0 = rel_poses[0]
    T_align = T_gt @ np.linalg.inv(T_pred0)  # Matrix multiplication
    # T_align = T_gt @ T_pred0

    # --- Apply to all poses ---
    os.makedirs(aligned_dir, exist_ok=True)

    for idx, (pose, path) in enumerate(zip(rel_poses, rel_pose_paths)):
        T_pred = pose
        T_aligned = T_align @ T_pred

        filename = os.path.basename(path)
        # filename = os.path.basename(f'ground_truth_{idx}.txt')
        out_path = os.path.join(aligned_dir, filename)
        np.savetxt(out_path, T_aligned, fmt="%.6f")
        print(f"✅ Saved aligned pose to {out_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Align pose sequences to ground truth")
    parser.add_argument('--poses_dir', type=str, required=True, help='Path to the folder containing frame-*.pose.txt and ground_truth_*.pose.txt')
    parser.add_argument('--aligned_dir', type=str, required=True, help='Path to the folder where aligned poses will be saved')
    
    args = parser.parse_args()
    main(args.poses_dir, args.aligned_dir)
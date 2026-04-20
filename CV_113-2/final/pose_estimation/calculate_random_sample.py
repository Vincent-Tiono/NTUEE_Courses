import cv2
import numpy as np
import os
import glob
from scipy.spatial.transform import Rotation as R
from scipy.linalg import svd
from scipy.optimize import least_squares
from concurrent.futures import ThreadPoolExecutor

import argparse
import random

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_pointcloud_alignment(xyz1_sampled, xyz2_sampled, label1, label2):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # xyz1_sampled: xyz1參考點（藍色）
    ax.scatter(
        xyz1_sampled[:, 0], xyz1_sampled[:, 1], xyz1_sampled[:, 2],
        c='blue', label=f'{label1}', s=20
    )

    # xyz2_sampled: xyz2參考點（紅色）
    ax.scatter(
        xyz2_sampled[:, 0], xyz2_sampled[:, 1], xyz2_sampled[:, 2],
        c='red', label=f'{label2}', s=20
    )

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Point Cloud Alignment')
    ax.legend()
    plt.tight_layout()
    plt.show()

def load_rgb_depth(rgb_path, depth_path):
    rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
    return rgb, depth

def depth_to_3d(depth_img, K):
    h, w = depth_img.shape
    i, j = np.meshgrid(np.arange(w), np.arange(h), indexing='xy')
    z = depth_img
    x = (i - K[0, 2]) * z / K[0, 0]
    y = (j - K[1, 2]) * z / K[1, 1]
    xyz = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    return xyz

def extract_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(gray, None)
    return kp, des

def match_features(des1, des2):
    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(des1, des2, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    return good_matches

def procrustes(A, B):
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B
    H = AA.T @ BB
    U, _, Vt = svd(H)
    R_ = Vt.T @ U.T
    if np.linalg.det(R_) < 0:
        Vt[-1, :] *= -1
        R_ = Vt.T @ U.T
    T_ = centroid_B.T - R_ @ centroid_A.T
    return R_, T_

# ======= Bundle Adjustment: Optimize R,T from 3D–2D matches =======
def project_points(X, R, T, K):
    X_cam = R @ X.T + T.reshape(3, 1)
    X_proj = K @ X_cam
    x = X_proj[:2] / X_proj[2:]
    return x.T

def reprojection_error(pose_vec, X, x_obs, K):
    rvec = pose_vec[:3]
    tvec = pose_vec[3:]
    R_mat, _ = cv2.Rodrigues(rvec)
    x_proj = project_points(X, R_mat, tvec, K)
    return (x_proj - x_obs).ravel()

def optimize_pose(R_init, T_init, X, x_obs, K):
    rvec_init, _ = cv2.Rodrigues(R_init)
    pose_vec = np.hstack([rvec_init.ravel(), T_init])
    res = least_squares(reprojection_error, 
                        pose_vec, args=(X, x_obs, K), 
                        loss="huber",
                        verbose=0)
    rvec_opt = res.x[:3]
    tvec_opt = res.x[3:]
    R_opt, _ = cv2.Rodrigues(rvec_opt)
    return R_opt, tvec_opt

def bilinear_depth_sampling(depth, pts):
    """
    depth: (H, W) 的深度圖 (float)
    pts:   (N, 2) 的浮點像素座標陣列，每列為 (u, v)
    回傳:  (N,) 的深度值 (float)，若超出邊界則給 0
    """
    H, W = depth.shape
    u = pts[:, 0]
    v = pts[:, 1]

    # 先算出最近的左上角整數座標 (x0, y0)，以及 (x1, y1) = (x0+1, y0+1)
    x0 = np.floor(u).astype(int)
    y0 = np.floor(v).astype(int)
    x1 = x0 + 1
    y1 = y0 + 1

    # 將邊界裁剪到合法範圍 [0, W-1], [0, H-1]
    x0_clipped = np.clip(x0, 0, W - 1)
    x1_clipped = np.clip(x1, 0, W - 1)
    y0_clipped = np.clip(y0, 0, H - 1)
    y1_clipped = np.clip(y1, 0, H - 1)

    # 取出四個鄰近像素的深度值
    Ia = depth[y0_clipped, x0_clipped]  # 左上
    Ib = depth[y1_clipped, x0_clipped]  # 左下
    Ic = depth[y0_clipped, x1_clipped]  # 右上
    Id = depth[y1_clipped, x1_clipped]  # 右下

    # 計算插值權重
    wa = (x1 - u) * (y1 - v)
    wb = (x1 - u) * (v - y0)
    wc = (u - x0) * (y1 - v)
    wd = (u - x0) * (v - y0)

    # 把權重乘上對應的深度
    depth_interp = wa * Ia + wb * Ib + wc * Ic + wd * Id

    # 如果原始 (u,v) 完全在影像外（即 u<0 or u>=W or v<0 or v>=H），就把深度設為 0
    outside = (u < 0) | (u >= W) | (v < 0) | (v >= H)
    depth_interp[outside] = 0.0

    return depth_interp

# ======= 主函式 =======
def compute_poses(rgb_list, depth_list, K, initial_pose_path):
    num_images = len(rgb_list)
    poses = [np.loadtxt(initial_pose_path)] + [np.zeros((1,)) for i in range(1, num_images)]
    rgb1, depth1 = load_rgb_depth(rgb_list[0], depth_list[0])
    xyz1 = depth_to_3d(depth1, K)
    kp1, des1 = extract_features(rgb1)

    err_list = [1e-5, 1e-4, 1e-3]

    for err in err_list:
        for i in range(1, num_images):
            if poses[i].ndim == 2: continue
            R_best = np.zeros((3, 3))
            T_best = np.zeros((3,))
            loss_min = 1e18
            pose_best = np.zeros((1,))
            for k in range(0, num_images):
                if poses[k].ndim < 2: continue
                if loss_min < err: continue

                rgb1, depth1 = load_rgb_depth(rgb_list[k], depth_list[k])
                xyz1 = depth_to_3d(depth1, K)
                kp1, des1 = extract_features(rgb1)

                rgb2, depth2 = load_rgb_depth(rgb_list[i], depth_list[i])
                xyz2 = depth_to_3d(depth2, K)
                kp2, des2 = extract_features(rgb2)
                matches = match_features(des1, des2)

                if len(matches) < 4: 
                    # print("skip")
                    continue

                pts1 = np.array([kp1[m.queryIdx].pt for m in matches])
                pts2 = np.array([kp2[m.trainIdx].pt for m in matches])

                # 插入 RANSAC 濾掉錯配
                E, mask_E = cv2.findEssentialMat(
                    pts1,
                    pts2,
                    K,
                    method=cv2.RANSAC,
                    prob=0.999,
                    threshold=1.0
                )

                # 篩掉非 inlier
                good_idx = (mask_E.ravel() == 1)
                if np.sum(good_idx) < 4:
                    # print("skip")
                    continue

                pts1 = pts1[good_idx]
                pts2 = pts2[good_idx]

                # 先用雙線性插值，在浮點像素 (u,v) 上算出深度 z1, z2 ---
                z1 = bilinear_depth_sampling(depth1, pts1)   # shape = (N,)
                z2 = bilinear_depth_sampling(depth2, pts2)   # shape = (N,)

                # 過濾掉深度為 0 的點 (invalid) ---
                valid = (z1 > 0) & (z2 > 0)

                # 篩出有效的像素座標與深度
                pts1_valid = pts1[valid]
                pts2_valid = pts2[valid]
                z1_valid   = z1[valid]
                z2_valid   = z2[valid]

                # 如果有效點太少，就跳過
                if len(z1_valid) < 4:
                    # print("skip")
                    continue

                # 由 (u,v,z) → (X,Y,Z)
                fx, fy = K[0, 0], K[1, 1]
                cx, cy = K[0, 2], K[1, 2]

                u1 = pts1_valid[:, 0]
                v1 = pts1_valid[:, 1]
                x1 = (u1 - cx) * z1_valid / fx
                y1 = (v1 - cy) * z1_valid / fy
                xyz1_sampled = np.stack((x1, y1, z1_valid), axis=1)

                u2 = pts2_valid[:, 0]
                v2 = pts2_valid[:, 1]
                x2 = (u2 - cx) * z2_valid / fx
                y2 = (v2 - cy) * z2_valid / fy
                xyz2_sampled = np.stack((x2, y2, z2_valid), axis=1)

                # print(min(8, len(xyz1_sampled)))
                for j in range(20):
                    selected_idx = random.sample(range(len(xyz1_sampled)), min(8, len(xyz1_sampled)))

                    selected_xyz1_sampled = xyz1_sampled[selected_idx]
                    selected_xyz2_sampled = xyz2_sampled[selected_idx]
                    
                    R_, T_ = procrustes(selected_xyz2_sampled, selected_xyz1_sampled)

                    # 加入 BA 優化
                    # R_opt, T_opt = optimize_pose(R_, T_, xyz2_sampled, pts2_valid, K)
                    R_opt, T_opt = optimize_pose(
                        R_, T_,
                        selected_xyz1_sampled,    # 改成第一張影像在第一張相機座標系下的 3D 點
                        pts2_valid[selected_idx],      # 仍然是對應到第二張影像的 2D 座標
                        K
                    )

                    selected_xyz1_aligned_opt = (R_opt @ selected_xyz1_sampled.T).T + T_opt

                    loss = np.mean((selected_xyz1_aligned_opt-selected_xyz2_sampled)**2)  # L2 norm
                    loss = loss / len(selected_idx)

                    if loss < loss_min:
                        loss_min = loss
                        R_best, T_best = R_opt, T_opt
                        H = np.eye(4)
                        H[:3, :3] = R_best
                        H[:3, 3] = T_best
                        if loss < err:
                            pose_best = poses[k] @ np.linalg.inv(H)

            
            if loss_min < err:
                # print(i, loss_min)
                poses[i] = pose_best

    return poses

K = np.array([
    [585.0, 0.0, 320.0],
    [0.0, 585.0, 240.0],
    [0.0, 0.0, 1.0]
])

def process_sparse_sequence(relative_path):
    # data_dir = os.path.join("../../7SCENES", relative_path)
    data_dir = relative_path
    print(data_dir)

    rgb_list = sorted(glob.glob(os.path.join(data_dir, "*.color.png")))
    depth_list = sorted(glob.glob(os.path.join(data_dir, "*.depth.proj.png")))
    initial_pose_path = os.path.join(data_dir, "frame-000000.pose.txt")

    if not os.path.exists(initial_pose_path):
        print(f"❌ 缺少初始姿態: {initial_pose_path}")
        return

    print(f"🧵 開始處理: {data_dir}")
    poses = compute_poses(rgb_list, depth_list, K, initial_pose_path)

    for i, pose in enumerate(poses):
        if i == 0 or pose.ndim < 2: continue
        filename = f"{data_dir}/{rgb_list[i][-23:-10]}.pose.txt"
        np.savetxt(filename, pose, fmt="%.6f")
    print(f"✅ 完成: {data_dir}")

def main():
    data_dirs = [
        "chess/test/seq-03",
        "fire/test/seq-03",
        "heads/test/seq-01",
        "office/test/seq-02",
        "office/test/seq-06",
        "office/test/seq-07",
        "office/test/seq-09",
        "pumpkin/test/seq-01",
        "redkitchen/test/seq-03",
        "redkitchen/test/seq-04",
        "redkitchen/test/seq-06",
        "redkitchen/test/seq-12",
        "redkitchen/test/seq-14",
        "stairs/test/seq-01"
    ]

    data_sparse_dirs = [
        "chess/test/sparse-seq-05",
        "fire/test/sparse-seq-04",
        "pumpkin/test/sparse-seq-07",
        "stairs/test/sparse-seq-04"
    ]

    parser = argparse.ArgumentParser(description="處理 dataset 的 sequence")
    parser.add_argument('--dataset', type=str, required=True, help='Dataset 根目錄的路徑')
    args = parser.parse_args()

    dataset_root = args.dataset
    # data_list = [os.path.join(dataset_root, d) for d in data_dirs]
    data_sparse_list = [os.path.join(dataset_root, d) for d in data_sparse_dirs]

    # 使用多執行緒處理
    # with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
    #     executor.map(process_sparse_sequence, data_sparse_list)
    
    # 不使用多執行緒處理
    for data in data_sparse_list:
        process_sparse_sequence(data)
    
    print("🎉 全部 sequence 處理完成")

if __name__ == "__main__":
    main()
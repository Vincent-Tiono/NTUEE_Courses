import os
import glob
import argparse

def main(pose_dir):
    # Get all .txt files in the directory
    all_txt_files = sorted(glob.glob(os.path.join(pose_dir, "*.txt")))

    # Delete all .txt files
    for f in all_txt_files:
        os.remove(f)
        print(f"🗑️ Deleted {f}")

    print(f"\n✅ Deleted {len(all_txt_files)} files in '{pose_dir}'.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Delete all .txt pose files in the specified directory.")
    parser.add_argument('--pose_dir', type=str, required=True, help='Directory containing pose .txt files')
    args = parser.parse_args()

    main(args.pose_dir)

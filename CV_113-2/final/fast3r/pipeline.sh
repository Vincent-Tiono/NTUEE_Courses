# 設定預設值
DEFAULT_BASE_DIR="/tmp2/vincentchang"
DEFAULT_DATASET="chess"
DEFAULT_SEQ_NUM="seq-03"

# 檢查參數，若無則使用預設值
BASE_DIR=${1:-$DEFAULT_BASE_DIR}
DATASET=${2:-$DEFAULT_DATASET}     # 使用的資料集名稱（如 chess）
SEQ_NUM=${3:-$DEFAULT_SEQ_NUM}     # 指定序列編號


# 顯示使用中的值
echo "Using base directory: $BASE_DIR"
echo "Using dataset: $DATASET"
echo "Using sequence: $SEQ_NUM"


# 設定其他變數
TEST_TRAIN=test                       # 使用訓練資料（train 或 test）
OUTPUT_PLY=${DATASET}-${SEQ_NUM}.ply  # 最終輸出的 PLY 檔名
POSE_DIR=poses                        # 儲存預測相機位姿的資料夾名稱
ALIGNED_DIR=aligned_poses             # 儲存對齊後相機位姿的資料夾名稱


# 建立必要的資料夾結構：預測位姿、對齊位姿及PLY輸出的目錄
if [ ! -d "./exp_data/$POSE_DIR" ]; then
    mkdir -p ./exp_data/$POSE_DIR/
    echo "Created directory: ./exp_data/$POSE_DIR/"
else
    echo "Directory already exists: ./exp_data/$POSE_DIR/"
fi

if [ ! -d "./exp_data/$ALIGNED_DIR" ]; then
    mkdir -p ./exp_data/$ALIGNED_DIR/
    echo "Created directory: ./exp_data/$ALIGNED_DIR/"
else
    echo "Directory already exists: ./exp_data/$ALIGNED_DIR/"
fi

if [ ! -d "./ply_output" ]; then
    mkdir -p ./ply_output/
    echo "Created directory: ./ply_output/"
else
    echo "Directory already exists: ./ply_output/"
fi

# 將 7SCENES 資料集中指定序列（$DATASET/$SEQ_NUM）之第 000000 幀的位姿檔
# 複製到實驗資料夾，並重新命名為 ground_truth_0.pose.txt 以作為對齊基準
cp $BASE_DIR/$DATASET/$TEST_TRAIN/$SEQ_NUM/frame-000000.pose.txt ./exp_data/$POSE_DIR/ground_truth_0.pose.txt

# 執行 Fast3R 批次預測，取得每張影像的相機位姿（pose），結果存到 ../exp_data/poses
python3 fast3r_test.py --image_dir $BASE_DIR/$DATASET/$TEST_TRAIN/$SEQ_NUM --output_dir ./exp_data/$POSE_DIR

# 將預測 pose 與 ground truth 對齊，輸出對齊後的 pose 到 ../exp_data/aligned_poses
python3 aligning.py --poses_dir ./exp_data/$POSE_DIR --aligned_dir ./exp_data/$ALIGNED_DIR

# 清空 $BASE_DIR/$DATASET/$SEQ_NUM/ 中的舊 pose 檔案（frame-XXXXXX.pose.txt）
python3 clean_file.py --pose_dir $BASE_DIR/$DATASET/$TEST_TRAIN/$SEQ_NUM

# 將對齊後的 pose 複製到 $BASE_DIR/$DATASET/$SEQ_NUM/ 中，準備匯出為 PLY
cp ./exp_data/$ALIGNED_DIR/* $BASE_DIR/$DATASET/$TEST_TRAIN/$SEQ_NUM/

# 執行 seq2ply，將影像序列及 pose 轉換成 PLY 點雲，輸出到 ./ply_output/
python3 seq2ply.py --seq_path $BASE_DIR/$DATASET/$TEST_TRAIN/$SEQ_NUM --output_name ./ply_output/$OUTPUT_PLY
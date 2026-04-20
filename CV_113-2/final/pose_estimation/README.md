## 參考環境設定

1.
```bash
conda create --name math python=3.11
conda activate math
```

2.
```bash
pip install opencv-python pillow open3d
```


## 執行指令列表

1. 
```bash
# Run dense dataset
python calculate.py --dataset <your_7SCENES_dataset_path>
python seq2ply.py --dataset <your_7SCENES_dataset_path>
```
2. 
```bash
# Run sparse dataset
python calculate_random_sample.py --dataset <your_7SCENES_dataset_path>
python seq2ply_sparse.py --dataset <your_7SCENES_dataset_path>
```

Or simply source run.sh after setting dataset path in run.sh:
```bash
source run.sh
```

## .ply files output

1. dense dataset
```bash
test\
```

1. sparse dataset
```bash
bonus\
```
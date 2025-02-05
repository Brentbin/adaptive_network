#!/bin/bash

# 生成数据
echo "Generating data..."
python src/data_gen.py

# 运行训练
echo "Training model..."
python src/train.py

# 分析结果
echo "Analyzing results..."
jupyter nbconvert --execute notebooks/analysis.ipynb --to html 
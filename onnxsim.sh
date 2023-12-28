#!/bin/bash

# 対象のフォルダを指定
target_folder="."

# 指定フォルダ内のすべての .onnx ファイルに対してループ
for file in "$target_folder"/*.onnx; do
    # onnxsim コマンドを実行
    onnxsim "$file" "$file"
done
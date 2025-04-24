#!/bin/bash

# Download the pretrained models for CogVideoX-2b and LanDiff

huggingface-cli download yinaoxiong/LanDiff --local-dir ./LanDiff

mkdir CogVideoX-2b-sat
cd CogVideoX-2b-sat
wget -c https://cloud.tsinghua.edu.cn/f/fdba7608a49c463ba754/?dl=1 -O vae.zip
unzip vae.zip
rm vae.zip
wget -c https://cloud.tsinghua.edu.cn/f/556a3e1329e74f1bac45/?dl=1 -O transformer.zip
unzip transformer.zip
rm transformer.zip
huggingface-cli download THUDM/CogVideoX-2b --include "tokenizer/*" --local-dir .
huggingface-cli download THUDM/CogVideoX-2b --include "text_encoder/*" --local-dir .
mkdir t5-v1_1-xxl
mv tokenizer/* text_encoder/* t5-v1_1-xxl
rm -rf tokenizer text_encoder
## Model Download

| Model       | Download Link                                                                                                                                       |           Download Link               |
|--------------|-----------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------|
| LanDiff      | 🤗 [Huggingface](https://huggingface.co/yinaoxiong/LanDiff)               | 🤖 [ModelScope](https://www.modelscope.cn/models/yinaoxiong/LanDiff)

To use LanDiff, you need to download the pre-trained model weights to the `ckpts` directory:

```bash
cd ckpts
# Download model weights
huggingface-cli download yinaoxiong/LanDiff --local-dir ./LanDiff
cd ..
```
or you can also download the model weights from [ModelScope](https://www.modelscope.cn/models/yinaoxiong/LanDiff).

```bash
cd ckpts
# Download model weights
modelscope download yinaoxiong/LanDiff --local_dir ./LanDiff
cd ..
```

The expected directory structure after downloading the weights should be:

```
ckpts
└── LanDiff
    ├── CogVideoX-2b-sat
    │   ├── t5-v1_1-xxl
    │   ├── transformer
    │   │   ├── 1000
    │   │   │   └── mp_rank_00_model_states.pt
    │   │   ├── latest
    │   └── vae
    │       ├── 3d-vae.pt
    ├── diffusion
    │   ├── 1
    │   │   └── mp_rank_00_model_states.pt
    │   └── latest
    ├── llm
    │   └── model.safetensors
    └── tokenizer
        └── model.safetensors
```

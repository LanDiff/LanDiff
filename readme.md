# LanDiff



## Environment Setup

### Prerequisites
- Python 3.10 or higher
- PyTorch 2.5 or higher

### Installation

```bash
# Create and activate Conda environment
conda create -n landiff python=3.10
conda activate landiff

# Install dependencies
pip install -r requirements.txt

# Install Flash Attention (specific version)
pip install flash-attn==2.7.4.post1 --no-build-isolation
```

## Model Weights

To use LanDiff, you need to download the pre-trained model weights to the `ckpts` directory:

```bash
# Create directory for model weights if it doesn't exist
mkdir -p ckpts

# Download model weights (replace with actual download commands)
wget -O ckpts/model_weights.pt https://example.com/path/to/weights
```

The expected directory structure after downloading the weights should be:

```
ckpts/
├── CogVideoX-2b-sat
└── LanDiff
    ├── diffusion
    │   ├── 1
    │   │   ├── mp_rank_00_model_states_origin.pt
    │   │   └── mp_rank_00_model_states.pt
    │   └── latest
    ├── llm
    │   └── model.safetensors
    └── tokenizer
        └── model.safetensors
```

## Usage Guide

### Basic Usage

```python
# Example code
export PYTHONPATH=$PYTHONPATH:$(pwd)
torchrun --standalone --nproc-per-node=1 landiff/infer_video.py --prompt "A snail with a brown and tan shell is seen crawling on a bed of green moss. The snail's body is grayish-brown, and it has two prominent tentacles extended forward. The environment suggests a natural, outdoor setting with a focus on the snail's movement across the mossy surface."
```

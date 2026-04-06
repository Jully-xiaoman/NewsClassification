from huggingface_hub import snapshot_download
import os

# 使用国内镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

snapshot_download(
    repo_id="google-bert/bert-base-chinese",
    local_dir="./model/bert-base-chinese",
    ignore_patterns=[
        ".gitattributes",
        "README.md",
        "*.msgpack",
        "*.onnx",
        "*.h5",
        "*.ot",
        "pytorch_model.bin"
    ]
)
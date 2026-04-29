# magiv3

conda create -n magiv3 python=3.10
conda activate magiv3

conda install -y numpy pillow matplotlib networkx shapely
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118
pip install transformers==4.49.0 einops timm pytorch-metric-learning
conda install ipykernel --update-deps --force-reinstall

pip install httpx openai
pip install bitsandbytes accelerate
pip install dill

# ------------------------------------------------------------
# llamacpp

conda create -n llamacpp python=3.10
conda activate llamacpp

sudo apt update
sudo apt install g++ build-essential

conda install -c conda-forge cuda-nvcc cuda-toolkit

git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
# cmake -B build -DGGML_CUDA=ON
cmake -B build -DGGML_CUDA=ON \
    -DCMAKE_INSTALL_RPATH="$CONDA_PREFIX/lib;$CONDA_PREFIX/targets/x86_64-linux/lib" \
    -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON
cmake --build build -j

# 启动服务器
conda activate llamacpp
cd ~/文档/AAvscode/llama.cpp
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$CONDA_PREFIX/targets/x86_64-linux/lib:$LD_LIBRARY_PATH

# 1. Qwen3.5-4B
./build/bin/llama-server \
    -m ./models/Qwen3.5-4B-IQ4_NL.gguf \
    --mmproj ./models/mmproj-F16-4B.gguf \
    --ctx-size 16384 \
    --port 8001 \
    --host 127.0.0.1

# 2. Qwen3.5-9B
./build/bin/llama-server \
    -m ./models/Qwen3.5-9B-Q6_K.gguf \
    --mmproj ./models/mmproj-F16-9B.gguf \
    --ctx-size 16384 \
    --port 8001 \
    --host 127.0.0.1
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

pip install opencc-python-reimplemented

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

# 0. Qwen3.5-0.8B
./build/bin/llama-server \
    -m ./models/Qwen3.5-0.8B-IQ4_NL.gguf \
    --mmproj ./models/mmproj-F16-0.8B.gguf \
    --ctx-size 8192 \
    --port 8001 \
    --host 127.0.0.1

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


# ------------------------------------------------------------
# 后端
pip install fastapi uvicorn python-multipart

# 前端
npx create-vite@5 frontend --template vue-ts
cd frontend
npm install
npm install element-plus vue-router@4 konva vue-konva axios
npm install vuedraggable@next


# uvicorn app.main:app --host 0.0.0.0 --port 8002 --reload
# npm run dev

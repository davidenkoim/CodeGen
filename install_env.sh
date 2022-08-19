# Install Anaconda
wget https://repo.anaconda.com/archive/Anaconda3-2021.11-Linux-x86_64.sh
bash ./Anaconda3-2021.11-Linux-x86_64.sh
rm ./Anaconda3-2021.11-Linux-x86_64.sh

# Create environment
conda init
conda create --name codeGen_env python=3.7
conda activate codeGen_env
conda config --add channels conda-forge
conda config --add channels pytorch

conda install pytorch torchvision torchaudio cudatoolkit=11.3
conda install six scikit-learn stringcase transformers ply slimit astunparse submitit
pip install transformers[onnx]
pip install cython
cd codegen_sources/model/tools
git clone https://github.com/glample/fastBPE.git

cd fastBPE
g++ -std=c++11 -pthread -O3 fastBPE/main.cc -IfastBPE -o fast
python setup.py install
cd ../../../../

mkdir tree-sitter
cd tree-sitter
git clone https://github.com/tree-sitter/tree-sitter-cpp.git
git clone https://github.com/tree-sitter/tree-sitter-java.git
git clone https://github.com/tree-sitter/tree-sitter-python.git
cd ..

cd codegen_sources/test_generation/
wget https://github.com/EvoSuite/evosuite/releases/download/v1.1.0/evosuite-1.1.0.jar
cd ../..

git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
cd ..

pip install sacrebleu=="1.2.11" javalang tree_sitter psutil fastBPE
pip install hydra-core --upgrade --pre
pip install black==19.10b0

# load dataset
mkdir ../datasets
cd ../datasets
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1ZVbo3spbPXQEF8GQACIl-QodaBKY-NDF' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1ZVbo3spbPXQEF8GQACIl-QodaBKY-NDF" -O java-med.zip && rm -rf /tmp/cookies.txt

module load gcc/6.3.0 python_gpu/3.7.1 cuda/10.1.243 cudnn/7.6.4 eth_proxy
source venv/bin/activate
pip3 install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
export PYTHONPATH=$PATHONPATH:~/reinforced-gnn/src

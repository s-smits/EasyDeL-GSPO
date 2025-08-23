Multi-worker GSPO on TPUs with EasyDeL 🔥

#PHASE 1
sudo apt update
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
sudo apt install tmux
tmux new -s tpu

#PHASE 2
uv venv .venv --python 3.11.13
source .venv/bin/activate
git clone https://github.com/s-smits/EasyDeL-GSPO.git
cd ~/EasyDeL-GSPO
git checkout math-only-improved
uv pip install -e .
uv pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
uv pip install -U transformers datasets wandb accelerate
uv pip install math-verify

#PHASE 3 (EACH WORKER, IN THIS EXAMPLE TPU-v4-16)
./run_gfspo_training.sh
# DL2

## MASt3R checkpoints

mkdir -p checkpoints/

wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth -P checkpoints/

wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth -P checkpoints/

wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl -P checkpoints/

## additional reqs install:
pip install scipy ftfy regex tqdm torch git+https://github.com/openai/CLIP.git einops pyrender==0.1.45 trimesh==3.9.34 pycollada==0.6

inside rubensklooisels git git clone https://github.com/peract/peract_colab.git
## RVT-2

For experiments on RLBench, we use [pre-generated dataset](https://drive.google.com/drive/folders/0B2LlLwoO3nfZfkFqMEhXWkxBdjJNNndGYl9uUDQwS1pfNkNHSzFDNGwzd1NnTmlpZXR1bVE?resourcekey=0-jRw5RaXEYRLe2W6aNrNFEQ) provided by [PerAct](https://github.com/peract/peract#download). Please download and place them under `RVT/rvt/data/test`.

Download the [pretrained RVT-2 model](https://huggingface.co/ankgoyal/rvt/tree/main/rvt2). Place the model (`model_99.pth` trained for 99 epochs or ~80K steps with batch size 192) and the config files under the folder `RVT/rvt/runs/rvt2/`. Run evaluation using (from folder `RVT/rvt`):

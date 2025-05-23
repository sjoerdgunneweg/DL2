# DL2

## MASt3R checkpoints

mkdir -p checkpoints/

wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth -P checkpoints/

wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth -P checkpoints/

wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl -P checkpoints/

## additional reqs install:
pip install scipy ftfy regex tqdm torch git+https://github.com/openai/CLIP.git einops pyrender==0.1.45 trimesh==3.9.34 pycollada==0.6

## RVT-2

This repository provides containerized options to run [RVT-2](https://github.com/nvlabs/rvt) for reproducibility and easier deployment. It supports both Docker and Singularity (Apptainer), including usage on HPC systems like Snellius.

### Setup Options

You can run RVT-2 in three different ways:

#### 1. Build Docker Image Locally
Use the provided Dockerfile to build the image:

```bash
docker build -t rvt2:latest .
```

#### 2. Use Prebuilt Docker Image
Pull and run the prebuilt image from [DockerHub](https://hub.docker.com/repository/docker/meeslindeman/rvt2/general).

```bash
docker build -t rvt2:latest .
```

#### 3. Build Singularity Image (HPC / Snellius)
Use the `rvt2build.def` file with Apptainer (formerly Singularity). For more information on using Singularity on Snellius, see [Surfdesk](https://servicedesk.surf.nl/wiki/spaces/WIKI/pages/30660251/Apptainer+formerly+Singularity).

### Environment Setup 
This should work automatically, make sure the following environment variables are set before running the container.

```bash
export COPPELIASIM_ROOT=<EDIT ME>/PATH/TO/COPPELIASIM/INSTALL/DIR
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
export DISPLAY=:1.0
```
### Data & Model Setup
The dataset and pretrained model are not included in the container and must be set up manually.

#### 1. Download Dataset
Download the [pre-generated dataset](https://drive.google.com/drive/folders/0B2LlLwoO3nfZfkFqMEhXWkxBdjJNNndGYl9uUDQwS1pfNkNHSzFDNGwzd1NnTmlpZXR1bVE?resourcekey=0-jRw5RaXEYRLe2W6aNrNFEQ) provided by [PerAct](https://github.com/peract/peract#download). Place the contents under the following directory structure:

```bash
RVT/rvt/data/
├── train/
├── val/
└── test/
```

#### 2. Download Pretrained Model
Download the [pretrained RVT-2 model](https://huggingface.co/ankgoyal/rvt/tree/main/rvt2). Place the model `model_99.pth` and the config files under the folder `RVT/rvt/runs/rvt2/`.

### Running the Container

#### Docker Example
Replace the paths with the location of your `data/` and `runs/` directories:

```bash
docker run --gpus all -it \
  -v /path/to/data:/root/install/RVT/rvt/data \
  -v /path/to/runs:/root/install/RVT/rvt/runs \
  -w /root/install/RVT/rvt \
  rvt2:latest
```

#### Singularity Example
Make sure to bind your local data and run directories, and use `--nv` for GPU support:

```bash
apptainer shell --nv \
  --bind /path/to/project:/rvt \
  --bind /path/to/data:/data \
  rvt2build.sif
```

### Evaluation
To reproduce RVT-2 evaluation on RLBench:

```bash
xvfb-run python eval.py \
  --model-folder runs/rvt2 \
  --eval-datafolder data/test \
  --tasks stack_cups \
  --eval-episodes 25 \
  --log-name test/1 \
  --device 0 \
  --headless \
  --model-name model_99.pth
```

Make sure paths match the bind mounts used in your Docker or Singularity command.

### Training or Evaluating Custom Models
To train or evaluate RVT-2 on new data or models, follow the same steps as above, but point to your own dataset and model paths when launching the container and running the evaluation script.

Additional instructions will be added here as the training process is finalized.

## RVT-2

For experiments on RLBench, we use [pre-generated dataset](https://drive.google.com/drive/folders/0B2LlLwoO3nfZfkFqMEhXWkxBdjJNNndGYl9uUDQwS1pfNkNHSzFDNGwzd1NnTmlpZXR1bVE?resourcekey=0-jRw5RaXEYRLe2W6aNrNFEQ) provided by [PerAct](https://github.com/peract/peract#download). Please download and place them under `RVT/rvt/data/test`.

Download the [pretrained RVT-2 model](https://huggingface.co/ankgoyal/rvt/tree/main/rvt2). Place the model (`model_99.pth` trained for 99 epochs or ~80K steps with batch size 192) and the config files under the folder `RVT/rvt/runs/rvt2/`. Run evaluation using (from folder `RVT/rvt`):

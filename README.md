# Enhancing Few-Shot Robotic Manipulation by Replacing Depth Sensors with Learned 3D Geometry from MASt3R

This project investigates a fundamental challenge in robotic manipulation: the dependency on expensive RGB-D sensors for accurate 3D spatial understanding. We explore whether [MASt3R](https://europe.naverlabs.com/blog/mast3r-matching-and-stereo-3d-reconstruction/) (Matching And Stereo 3D Reconstruction), a 3D reconstruction method that works with RGB images alone, can effectively replace traditional depth sensors in robotic manipulation pipelines, by integrating depthmaps created using MASt3R with [RVT-2](https://robotic-view-transformer-2.github.io/) (Robotic Vision Transformer).

**Authors:** Ruben Figge, Sjoerd Gunneweg, Jurgen de Heus, Mees Lindeman  
**University of Amsterdam**

## Table of Contents

- [Strengths, Weaknesses, and Motivation](#strengths-weaknesses-and-motivation)  
- [Novel Contributions](#novel-contributionas)  
- [Results](#results)  
  * [MASt3R Reproducibility (DTU Dataset)](#mast3r-reproducibility-dtu-dataset)  
  * [Depth Map Quality Assessment (RLBench)](#depth-map-quality-assessment-rlbench)  
  * [RVT-2 Integration Results](#rvt-2-integration-results)  
- [Conclusion](#conclusion)  
- [Description of Each Student's Contribution](#description-of-each-students-contribution)  
- [Technical Setup](#technical-setup)  
  * [MASt3R](#mast3r)  
  * [Benchmarking MASt3R on RLBench Depth Estimation](#benchmarking-mast3r-on-rlbench-depth-estimation)  
  * [MASt3R Checkpoints](#mast3r-checkpoints)  
  * [RVT-2](#rvt-2)  
  * [Setup Options](#setup-options)  
  * [Environment Setup](#environment-setup)  
  * [Data & Model Setup](#data--model-setup)  
  * [Running the Container](#running-the-container)  
  * [Evaluation](#evaluation)  
  * [Training or Evaluating Custom Models](#training-or-evaluating-custom-models)  
  * [Training from Scratch on MASt3R-Generated Data](#training-from-scratch-on-mast3r-generated-data)  

## Strengths and Weaknesses

### Strengths of MASt3R
- **Hardware Independence:** Eliminates the need for expensive RGB-D sensors and complex calibration procedures
- **Robustness:** Demonstrates strong performance under large baselines and low-texture scenes

### Identified Weaknesses
- **Reproducibility Challenges:** Limited documentation and missing evaluation scripts in the original implementation
- **Domain Gap:** Unclear how well laboratory benchmark performance translates to real robotic scenarios
- **Integration Complexity:** No existing framework for incorporating MASt3R outputs into manipulation pipelines

## Novel contributions

This project introduces two primary novel contributions to the field of vision-based robotic manipulation:

1. Benchmarking MASt3R on Synthetic Robotic Data <br>
We provide an evaluation of depth maps generated using the MASt3R model on the RLBench dataset. This dataset consists of synthetic, low-resolution, multi-view robotic scenes. While MASt3R has been trained on synthetic data, it has only been benchmarked on real-world datasets such as DTU; its performance on synthetic scenes remained unexplored. Our benchmarking against ground-truth RLBench depth maps extends the evaluation of MASt3R from real-world to simulated domains, revealing insights into its generalizability under reduced visual complexity, such as limited shadows, reflections, and texture variance.

2. Generalizing the RVT-2 Pipeline by Using Estimated Depth Maps <br>
The core contribution of this work lies in modifying the RVT-2 pipeline to eliminate the need for RGB-D input. We achieve this by integrating MAST3R into the pipeline as a plug-and-play depth estimation backbone, enabling RVT-2 to become a purely RGB-based system. To the best of our knowledge, this is the first demonstration of RVT-2 functioning without ground-truth depth information, relying instead on learned 3D geometry from uncalibrated RGB inputs. This significantly reduces hardware requirements, offering a viable path towards scalable and sensor-light robotic training systems.

In combination, these contributions bridge the gap between state-of-the-art 3D reconstruction methods and practical robotic control pipelines, paving the way for future systems that learn manipulation skills in sensor-limited environments.

## Results

### MASt3R Reproducibility (DTU Dataset)

### Depth Map Quality Assessment (RLBench)

### RVT-2 Integration Results

## Reproducing Results

### MASt3R

### Benchmarking MASt3R on RLBench depth estimation
To reproduce the results of the benchmarking of the depth estimation using MASt3R on the RLBench dataset the following steps can be followed.
#### 1. Using MAST3R to predict depthmaps using the RLBench rgb images
RVT-2 uses only the following cameras: front,
left shoulder, right shoulder, and wrist. For this reason we only predict the depthmaps of these camera views. 

augmenting_rlbench_with_mast3r/augment_rlbench_with_mast3r.py is used for this.
##### Example used for reproducing the paper results
```bash
cd augmenting_rlbench_with_mast3r

python augment_rlbench_with_mast3r.py --data_path ../../RLBench/data/train (TODO check path) --tasks close_jar insert_onto_square_peg open_drawer push_buttons

python augment_rlbench_with_mast3r.py --data_path ../../RLBench/data/val (TODO check path) --tasks close_jar insert_onto_square_peg open_drawer push_buttons
```

(By default when no tasks are specified all tasks inside the provided data folder are processed.)
Note that the predicted depthmaps are saved using the same method as the RLBench depthmaps making the images not interpretable without correct processing during loading them. So do not worry when they are not looking like much.

#### 2. Evaluating the estimated depthmaps against the ground-truth depthmaps provided by RLBench
For evaluating the depthmaps the metrics mentioned in the paper are used on each of the depthmaps. The metrics are both aggregated on episode level and task level. The metrics of each timestep is stored in a structured way together with the episode and task aggregates inside a json.

augmenting_rlbench_with_mast3r/evaluate_depth_maps.py is used for this.
##### Example used for reproducing the paper results
```bash
cd augmenting_rlbench_with_mast3r

python evaluate_depth_maps.py --data_path ../../RLBench/data/train (TODO check path) --tasks close_jar insert_onto_square_peg open_drawer push_buttons --outpath train_metrics.json 

python evaluate_depth_maps.py --data_path ../../RLBench/data/val (TODO check path) --tasks close_jar insert_onto_square_peg open_drawer push_buttons --outpath val_metrics.json 
```


### MASt3R checkpoints

mkdir -p checkpoints/

wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth -P checkpoints/

wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth -P checkpoints/

wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl -P checkpoints/

## additional reqs install:
pip install scipy ftfy regex tqdm torch git+https://github.com/openai/CLIP.git einops pyrender==0.1.45 trimesh==3.9.34 pycollada==0.6 
pip install scikit-image


### RVT-2

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
To train or evaluate RVT-2 on new data or models, follow the same steps as above, but point to your own dataset and model paths when launching the container and running the evaluation script. We also support training RVT-2 on pointmaps generated by MASt3R. Our pretrained models on MASt3R-generated data are hosted at:

- [HuggingFace Repository](https://huggingface.co/meeslindeman/rvt-2/tree/main/mast3r)

### Training from Scratch on MASt3R-Generated Data

To train RVT-2 using MASt3R-generated pointmaps, follow these steps:

1. **Prepare your data directory**
   - Store your pointmaps in the same directory structure as `rvt/data/{train, val, test}`.

2. **Store your config and checkpoints**
   - Training logs and model checkpoints will be saved in a user-defined directory (e.g. `rvt/runs/mast3r/`).

3. **Create a Training Job Script**

You can create a job script like the one below (example for Snellius using Apptainer):

```bash

```

## Conclusion

## Description of each students contribution

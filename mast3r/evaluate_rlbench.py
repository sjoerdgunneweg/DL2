import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from mast3r.model import AsymmetricMASt3R
from mast3r.fast_nn import fast_reciprocal_NNs

import mast3r.utils.path_to_dust3r
from dust3r.inference import inference
from dust3r.utils.image import load_images, rgb
from dust3r.image_pairs import make_pairs
from dust3r.utils.device import to_numpy
from dust3r.demo import get_3D_model_from_scene
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode


if __name__ == '__main__':
    device = 'cuda'
    schedule = 'cosine'
    lr = 0.01
    niter = 300


    # Load the model:
    model_name = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
    model = AsymmetricMASt3R.from_pretrained(model_name).to(device)


    # Load the dataset:
    
    images = load_images(['../img1.png', '../img2.png', '../img3.png'], size=512, square_ok=True)
    pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
    output = inference([tuple(images)], model, device, batch_size=1, verbose=False)

    mode = GlobalAlignerMode.PointCloudOptimizer if len(images) > 2 else GlobalAlignerMode.PairViewer
    scene = global_aligner(output, device=device, mode=mode, verbose=False)
    
    rgbimg = scene.imgs
    depths = to_numpy(scene.get_depthmaps())

    imgs = []
    for i in range(len(rgbimg)):
        imgs.append(rgb(depths[i]))

    # plot the depth of image 1 using 3d points:depth

    plt.figure()
    depth = Image.fromarray(imgs[0])
    depth = depth.resize((128, 128), Image.LANCZOS)
    plt.imshow(np.asarray(depth))
    plt.savefig('test.pdf')
    plt.show(block=True)


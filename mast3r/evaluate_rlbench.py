import matplotlib.pyplot as plt
import numpy as np

from mast3r.model import AsymmetricMASt3R
from mast3r.fast_nn import fast_reciprocal_NNs

import mast3r.utils.path_to_dust3r
from dust3r.inference import inference
from dust3r.utils.image import load_images

if __name__ == '__main__':
    device = 'cuda'
    schedule = 'cosine'
    lr = 0.01
    niter = 300


    # Load the model:
    model_name = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
    model = AsymmetricMASt3R.from_pretrained(model_name).to(device)


    # Load the dataset:
    
    images = load_images(['../img1.png', '../img2.png'], size=128, square_ok=True)
    output = inference([tuple(images)], model, device, batch_size=1, verbose=False)

    plt.figure()
    plt.imshow(np.transpose(images[0]["img"][0], (1, 2, 0)))
    plt.savefig('test.png')
    plt.show(block=True)

    # at this stage, you have the raw dust3r predictions
    view1, pred1 = output['view1'], output['pred1']
    view2, pred2 = output['view2'], output['pred2']

    pnts3d1 = pred1["pts3d"]
    pnts3d2 = pred2["pts3d_in_other_view"]

    # plot the depth of image 1 using 3d points:depth

    plt.figure()
    plt.imshow(pnts3d1[0, :, :, 2].cpu().numpy())
    plt.savefig('test.pdf')
    plt.show(block=True)
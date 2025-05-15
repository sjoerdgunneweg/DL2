import torchvision.transforms as tvf
from PIL import Image
import numpy as np
import torch

ImgNorm = tvf.Compose([
    tvf.ToTensor(),
    tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


def rgb(ftensor, true_shape=None):
    if isinstance(ftensor, list):
        return [rgb(x, true_shape=true_shape) for x in ftensor]
    if isinstance(ftensor, torch.Tensor):
        ftensor = ftensor.detach().cpu().numpy()  # H,W,3
    if ftensor.ndim == 3 and ftensor.shape[0] == 3:
        ftensor = ftensor.transpose(1, 2, 0)
    elif ftensor.ndim == 4 and ftensor.shape[1] == 3:
        ftensor = ftensor.transpose(0, 2, 3, 1)
    if true_shape is not None:
        H, W = true_shape
        ftensor = ftensor[:H, :W]
    if ftensor.dtype == np.uint8:
        img = np.float32(ftensor) / 255
    else:
        img = (ftensor * 0.5) + 0.5
    return img.clip(min=0, max=1)


def _resize_pil_image(img, long_edge_size):
    S = max(img.size)
    interp = Image.LANCZOS if S > long_edge_size else Image.BICUBIC
    new_size = tuple(int(round(x * long_edge_size / S)) for x in img.size)
    return img.resize(new_size, interp)

def load_images_from_loaded(images, size, square_ok=False, verbose=True):
    """
    Process a list of already loaded images (as PIL.Image or NumPy arrays)
    and convert them to normalized tensors for DUSt3R.
    """
    if verbose:
        print(f'>> Processing {len(images)} already-loaded images')

    imgs = []
    for idx, img in enumerate(images):
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        elif not isinstance(img, Image.Image):
            raise TypeError(f"Unsupported image type at index {idx}: {type(img)}")

        img = img.convert('RGB')  # ensure RGB
        W1, H1 = img.size

        # Resize
        if size == 224:
            img = _resize_pil_image(img, round(size * max(W1/H1, H1/W1)))
        else:
            img = _resize_pil_image(img, size)

        # Center crop
        W, H = img.size
        cx, cy = W // 2, H // 2
        if size == 224:
            half = min(cx, cy)
            img = img.crop((cx - half, cy - half, cx + half, cy + half))
        else:
            halfw, halfh = ((2 * cx) // 16) * 8, ((2 * cy) // 16) * 8
            if not square_ok and W == H:
                halfh = int(3 * halfw / 4)
            img = img.crop((cx - halfw, cy - halfh, cx + halfw, cy + halfh))

        W2, H2 = img.size
        if verbose:
            print(f' - processed image {idx} with resolution {W1}x{H1} --> {W2}x{H2}')

        imgs.append(dict(
            img=ImgNorm(img)[None],
            true_shape=np.int32([img.size[::-1]]),
            idx=idx,
            instance=str(idx)
        ))

    assert imgs, 'No valid images were processed'
    if verbose:
        print(f' (Successfully processed {len(imgs)} images)')
    return imgs



import mast3r.mast3r.utils.path_to_dust3r
from dust3r.inference import inference
from dust3r.image_pairs import make_pairs
from dust3r.utils.device import to_numpy
from dust3r.demo import get_3D_model_from_scene
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

def create_depth_maps(np_img_list, model, device, batch_size=1, niter=300, schedule='cosine', lr=0.01):
    images = load_images_from_loaded(np_img_list, size=512, square_ok=True)
            
    pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
    output = inference(pairs, model, device, batch_size=1, verbose=False)

    mode = GlobalAlignerMode.PointCloudOptimizer #if len(images) > 2 else GlobalAlignerMode.PairViewer
    scene = global_aligner(output, device=device, mode=mode, verbose=False)
    if mode == GlobalAlignerMode.PointCloudOptimizer:
        loss = scene.compute_global_alignment(init='mst', niter=niter, schedule=schedule, lr=lr)

    rgbimg = scene.imgs
    depths = to_numpy(scene.get_depthmaps())
    depths_max = max([d.max() for d in depths])
    depths = [d / depths_max for d in depths]
    print(depths[0])

    imgs = []
    for i in range(len(rgbimg)):
        imgs.append(rgb(depths[i]))

    return imgs
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



import mast3r.utils.path_to_dust3r
from dust3r.inference import inference
from dust3r.image_pairs import make_pairs
from dust3r.utils.device import to_numpy
from dust3r.demo import get_3D_model_from_scene
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

def create_depth_maps(np_img_list, model, device, batch_size=1, niter=150, schedule='cosine', lr=0.01):
    images = load_images_from_loaded(np_img_list, size=512, square_ok=True, verbose=False)
            
    pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
    output = inference(pairs, model, device, batch_size=1, verbose=False)

    mode = GlobalAlignerMode.PointCloudOptimizer #if len(images) > 2 else GlobalAlignerMode.PairViewer
    scene = global_aligner(output, device=device, mode=mode, verbose=False)
    if mode == GlobalAlignerMode.PointCloudOptimizer:
        loss = scene.compute_global_alignment(init='mst', niter=niter, schedule=schedule, lr=lr)

    rgbimg = scene.imgs
    depths = to_numpy(scene.get_depthmaps())

    imgs = []
    for i in range(len(rgbimg)):
        imgs.append(np.array(Image.fromarray(depths[i]).resize((128, 128), Image.LANCZOS)))

    return imgs








# Code copied from RLBench:  https://github.com/stepjam/RLBench/blob/master/rlbench/backend/utils.py
def ClipFloatValues(float_array, min_value, max_value):
  """Clips values to the range [min_value, max_value].

  First checks if any values are out of range and prints a message.
  Then clips all values to the given range.

  Args:
    float_array: 2D array of floating point values to be clipped.
    min_value: Minimum value of clip range.
    max_value: Maximum value of clip range.

  Returns:
    The clipped array.

  """
  if float_array.min() < min_value or float_array.max() > max_value:
    float_array = np.clip(float_array, min_value, max_value)
  return float_array



DEFAULT_RGB_SCALE_FACTOR = 256000.0

def float_array_to_rgb_image(float_array,
                             scale_factor=DEFAULT_RGB_SCALE_FACTOR,
                             drop_blue=False):
  """Convert a floating point array of values to an RGB image.

  Convert floating point values to a fixed point representation where
  the RGB bytes represent a 24-bit integer.
  R is the high order byte.
  B is the low order byte.
  The precision of the depth image is 1/256 mm.

  Floating point values are scaled so that the integer values cover
  the representable range of depths.

  This image representation should only use lossless compression.

  Args:
    float_array: Input array of floating point depth values in meters.
    scale_factor: Scale value applied to all float values.
    drop_blue: Zero out the blue channel to improve compression, results in 1mm
      precision depth values.

  Returns:
    24-bit RGB PIL Image object representing depth values.
  """
  # Scale the floating point array.
  scaled_array = np.floor(float_array * scale_factor + 0.5)
  # Convert the array to integer type and clip to representable range.
  min_inttype = 0
  max_inttype = 2**24 - 1
  scaled_array = ClipFloatValues(scaled_array, min_inttype, max_inttype)
  int_array = scaled_array.astype(np.uint32)
  # Calculate:
  #   r = (f / 256) / 256  high byte
  #   g = (f / 256) % 256  middle byte
  #   b = f % 256          low byte
  rg = np.divide(int_array, 256)
  r = np.divide(rg, 256)
  g = np.mod(rg, 256)
  image_shape = int_array.shape
  rgb_array = np.zeros((image_shape[0], image_shape[1], 3), dtype=np.uint8)
  rgb_array[..., 0] = r
  rgb_array[..., 1] = g
  if not drop_blue:
    # Calculate the blue channel and add it to the array.
    b = np.mod(int_array, 256)
    rgb_array[..., 2] = b
  image_mode = 'RGB'
  image = Image.fromarray(rgb_array, mode=image_mode)
  return image
from mast3r.model import AsymmetricMASt3R
from dust3r.utils.image import load_images
from dust3r.inference import inference

model = AsymmetricMASt3R.from_pretrained("naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric").cuda()
images = load_images(["/home/scur2628/DL2/mast3r/dust3r/croco/assets/Chateau1.png", "/home/scur2628/DL2/mast3r/dust3r/croco/assets/Chateau2.png"], size=512)
out = inference([tuple(images)], model, device="cuda", batch_size=1)

print("✅ Success! Inference ran and returned keys:", out.keys())

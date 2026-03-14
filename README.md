Real-ESRGAN  
Step-by-Step Usage Guide & Model Reference
Google Colab 
AI-Powered Image Upscaling: Enhancing Low-Resolution Imagery Using Real-ESRGAN x4plus
 Step-by-Step Usage Guide

1	Install
Run once at the start of every session before anything else.
!pip install realesrgan -q

2	Apply Compatibility Patch  (Critical — run before imports)
basicsr imports a module removed in torchvision v0.17+. This patch fixes it in memory — no reinstalls required.
import sys, types
import torchvision.transforms.functional as F

mod = types.ModuleType("torchvision.transforms.functional_tensor")
mod.rgb_to_grayscale = F.rgb_to_grayscale
sys.modules["torchvision.transforms.functional_tensor"] = mod

3	Import Libraries
Load all required modules after the patch is applied.
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
import cv2, urllib.request, os

4	Download Model Weights
~64 MB download from GitHub. Cached after first run.
os.makedirs("weights", exist_ok=True)

model_url  = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
model_path = "weights/RealESRGAN_x4plus.pth"

if not os.path.exists(model_path):
    urllib.request.urlretrieve(model_url, model_path)
    print("Downloaded!")

5	Load the Model
Initialise the RRDB network and the upsampler.
model = RRDBNet(
    num_in_ch=3,   num_out_ch=3,
    num_feat=64,   num_block=23,
    num_grow_ch=32, scale=4
)

upsampler = RealESRGANer(
    scale=4,
    model_path='weights/RealESRGAN_x4plus.pth',
    model=model,
    device='cuda'   # use 'cpu' if no GPU
)

6	Upload Your Image
Use Colab's file picker or set the path manually.
# Option A — Colab upload widget
from google.colab import files
uploaded = files.upload()
input_filename = list(uploaded.keys())[0]

# Option B — manual path
# input_filename = 'input.jpg'


7	Run Upscaling
Enhance the image and save the output. Confirmed working.
img = cv2.imread('input.jpg', cv2.IMREAD_UNCHANGED)

output, _ = upsampler.enhance(img, outscale=4)

cv2.imwrite('output.jpg', output)
print("Done! Saved to output.jpg")


8	Preview & Download
View result in notebook and download to your machine.
from IPython.display import display
from PIL import Image
from google.colab import files

display(Image.open('input.jpg').resize((400,300)))
display(Image.open('output.jpg').resize((400,300)))

files.download('output.jpg')

Results
Test 1
Before

 
After
 



Test 2 
Before
 
After
 



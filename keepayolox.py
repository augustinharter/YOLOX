#%%
import argparse
from cgi import test
from dataclasses import replace
import os
from matplotlib import pyplot as plt
import torch
import numpy as np
from PIL import Image
from torchvision.transforms import Normalize
from torch.utils.mobile_optimizer import optimize_for_mobile
import torchvision
os.environ["PYTHONPATH"] = os.getcwd()
from yolox.exp.build import get_exp
# import bounding boxes drawer
from yolox.utils import vis
from yolox.data.datasets import COCO_CLASSES
torch.set_printoptions(precision=2)
#%%
def custompostprocesssingle(prediction, num_classes, conf_thre=0.7, nms_thre=0.45, class_agnostic=False):
    box_corner = prediction.new(prediction.shape).detach()
    box_corner[:, 0] = prediction[:, 0] - prediction[:, 2] / 2
    box_corner[:, 1] = prediction[:, 1] - prediction[:, 3] / 2
    box_corner[:, 2] = prediction[:, 0] + prediction[:, 2] / 2
    box_corner[:, 3] = prediction[:, 1] + prediction[:, 3] / 2
    prediction[:, :4] = box_corner[:, :4]

    # If none are remaining => process next image
    image_pred = prediction
    # Get score and class with highest confidence
    #class_conf, class_pred = torch.max(image_pred[:, 5: 5 + num_classes], 1, keepdim=True)

    conf_mask = (image_pred[:, 4] >= conf_thre).squeeze()
    #conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thre).squeeze()
    # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
    #detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)
    #detections = detections[conf_mask]
    detections = image_pred[conf_mask]

    if class_agnostic:
        nms_out_index = torchvision.ops.nms(
            detections[:, :4],
            detections[:, 4],# * detections[:, 5],
            nms_thre,
        )
    else:
        nms_out_index = torchvision.ops.batched_nms(
            detections[:, :4],
            detections[:, 4],# * detections[:, 5],
            detections[:, 6],
            nms_thre,
        )

    detections = detections[nms_out_index]
    return detections

#%%

modeltype = "tiny"
outputname = "yolox.ptl"
ckpt_file = f'{"yolox_"+modeltype}.pth'
name = "yolox-"+modeltype
#%%
# demo 
!python tools/demo.py image -n {name} -c {ckpt_file} --save_result \
    --path /home/augo/coding/keepa/ml/automl/efficientdet/carsandbikes416.png
#%%
!python tools/export_torchscript.py --output-name {outputname} -n {name} -c {ckpt_file}\
    --decode_in_inference

#%% LOAD NORMAL MODEL
# load the model state dict
exp = get_exp(None, name)
model = exp.get_model()
print(model)
ckpt = torch.load(ckpt_file, map_location="cpu")

model.eval()
if "model" in ckpt:
    ckpt = ckpt["model"]
model.load_state_dict(ckpt)
#model.head.decode_in_inference = args.decode_in_inference
#%% CONVERT TO JIT
nmsthresh = 0.2
testconf = 0.01
class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forwardold(self, x):
        out = model(x)[0]
        nms_out_index = torchvision.ops.nms(
                out[:, :4],
                out[:, 4],
                0.45,
            )
        out = out[nms_out_index]
        return out[None]

    def forward(self, x):
        out = self.model(x)[0]
        out = custompostprocesssingle(out, exp.num_classes, testconf, nmsthresh, True)
        return out

print('NUM CLASSES, TEST CONF, NMSTHRESH',exp.num_classes, exp.test_conf, exp.nmsthre)
wrapper = Wrapper(model)

dummy_input = torch.randn(1, 3, exp.test_size[0], exp.test_size[1]).detach()

jitmod = torch.jit.trace(wrapper, dummy_input)

#%% OPTIMIZE FOR MOBILE
optimized_traced_model = optimize_for_mobile(jitmod)
outpath = f"/home/augo/coding/keepa/app/keepa_auth/assets/models/{outputname}"
optimized_traced_model._save_for_lite_interpreter(outpath)
print("generated torchscript mobile model at {}".format(outpath))
mobilemod = optimized_traced_model
#%%
# load jit model
modelpath = f"/home/augo/coding/keepa/app/keepa_auth/assets/models/{outputname}"
mobilemod = torch.jit.load(modelpath)
mobilemod.eval()
#%%
# load test image
size = 416
imgpath = "/home/augo/coding/keepa/ml/automl/efficientdet/carsandbikes.png"
imgpath416 = "/home/augo/coding/keepa/ml/automl/efficientdet/carsandbikes416.png"
img = Image.open(imgpath)
img = img.resize((size, size))
img.save(imgpath416)
plt.imshow(img)
imgTensor = torch.Tensor(np.array(img))
imgTensor = imgTensor.permute(2, 0, 1)[None]
normalize = Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
#imgTensor = normalize(imgTensor)

print('img after normalize:', 
    imgTensor.shape, '\n max:', 
    imgTensor.max(), '\n min:', 
    imgTensor.min(), '\n std:',
    imgTensor[0].flatten(1).std(), '\n mean:',
    imgTensor.flatten(1).mean())
print(imgTensor.shape)
#%%
# RUN MODELS
def run(mod):
    out = mod(imgTensor)[0]
    out.shape
    print('outputs first 3 10','\n', out.shape,'\n', out[:3,:10].detach().numpy())
    return out
    
outs = (
    custompostprocesssingle(model(imgTensor)[0], 80, testconf, nmsthresh, True), 
    jitmod(imgTensor), 
    mobilemod(imgTensor)
    )

print([out.shape for out in outs])

#%% VISUALIZE
def vis_wrap(out):
    boxImg = vis(np.array(img), out[:,0:4], out[:,4], torch.zeros_like(out[:,6]), conf=0.3, class_names=("",)+COCO_CLASSES)
    plt.imshow(boxImg)
    plt.show()

vis_wrap(outs[0])
vis_wrap(outs[1])
vis_wrap(outs[2])
# %%
# %% DIRECT VIS
preds = mobilemod(imgTensor)
print(preds.shape)
boxImg = vis(np.array(img), preds[:,0:4], preds[:,4]*preds[:,5], torch.ones_like(preds[:,4]), conf=0.3, class_names=["","car", "bike"])
plt.imshow(boxImg)
plt.show()
# %%

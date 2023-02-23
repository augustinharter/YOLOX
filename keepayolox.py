# %%
import sklearn.neighbors as neighbors
from pycocotools.coco import COCO
from time import time
from timeit import timeit
from torchvision.ops import box_iou
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from yolox.data.datasets import COCO_CLASSES
from yolox.utils import vis
from yolox.exp.build import get_exp
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
# import bounding boxes drawer
torch.set_printoptions(precision=2)
# %%


def getBoxCornersFromCOCOBox(box):
    return (box[0], box[1], box[0]+box[2], box[1]+box[3])


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
    class_conf, class_pred = torch.max(image_pred[:, 5: 5 + num_classes], 1, keepdim=True)

    conf_mask = (image_pred[:, 4] >= conf_thre).squeeze()
    #conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thre).squeeze()
    # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
    detections = torch.cat((image_pred[:, :5], class_conf,
                           class_pred.float(), image_pred[:, 5 + num_classes:]), 1)
    detections = detections[conf_mask]
    #detections = image_pred[conf_mask]

    if class_agnostic:
        nms_out_index = torchvision.ops.nms(
            detections[:, :4],
            detections[:, 4],  # * detections[:, 5],
            nms_thre,
        )
    else:
        nms_out_index = torchvision.ops.batched_nms(
            detections[:, :4],
            detections[:, 4],  # * detections[:, 5],
            detections[:, 6],
            nms_thre,
        )

    detections = detections[nms_out_index]
    return detections

# %%


modeltype = "tiny"
outputname = "yolox.ptl"
ckpt_file = f'{"yolox_"+modeltype}.pth'
name = "yolox-"+modeltype
'''
# demo
!python tools/demo.py image - n {name} - c {ckpt_file} - -save_result \
    - -path / home/augo/coding/keepa/ml/automl/efficientdet/carsandbikes416.png
!python tools/export_torchscript.py - -output-name {outputname} - n {name} - c {ckpt_file}\
    - -decode_in_inference
'''

# %% LOAD NORMAL MODEL
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
# %% CONVERT TO JIT
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


print('NUM CLASSES, TEST CONF, NMSTHRESH', exp.num_classes, exp.test_conf, exp.nmsthre)
wrapper = Wrapper(model)

dummy_input = torch.randn(1, 3, exp.test_size[0], exp.test_size[1]).detach()

jitmod = torch.jit.trace(wrapper, dummy_input)
jitmod.eval()

# %% OPTIMIZE FOR MOBILE
optimized_traced_model = optimize_for_mobile(jitmod)
outpath = f"/home/augo/coding/keepa/app/keepa_auth/assets/models/{outputname}"
optimized_traced_model._save_for_lite_interpreter(outpath)
print("generated torchscript mobile model at {}".format(outpath))
mobilemod = optimized_traced_model
mobilemod.eval()
# %%
# load jit model
modelpath = f"/home/augo/coding/keepa/app/keepa_auth/assets/models/{outputname}"
mobilemod = torch.jit.load(modelpath)
mobilemod.eval()
# %%
# load test image
size = 416
imgpath = "/home/augo/coding/keepa/ml/automl/efficientdet/carsandbikes.png"
imgpath416 = "/home/augo/coding/keepa/ml/automl/efficientdet/carsandbikes416.png"
pilimg = Image.open(imgpath)
img = pilimg.resize((size, size))
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
# %% VISUALIZE


def vis_wrap(out, img, confthresh=0.3, showscore=True):
    boxImg = vis(np.ascontiguousarray(img, dtype=np.uint8), out[:, 0:4], out[:, 4],  # *out[:,5],
                 out[:, 6], conf=confthresh, class_names=COCO_CLASSES, showscore=showscore)
    plt.imshow(boxImg)
    plt.show()


# %%
# RUN MODELS
'''
def run(mod):
    out = mod(imgTensor)[0]
    out.shape
    print('outputs first 3 10', '\n', out.shape, '\n', out[:3, :10].detach().numpy())
    return out


outs = (
    custompostprocesssingle(model(imgTensor)[0], 80, testconf, nmsthresh, True),
    jitmod(imgTensor),
    mobilemod(imgTensor)
)

print([out.shape for out in outs])


vis_wrap(outs[0], img)
vis_wrap(outs[1], img)
vis_wrap(outs[2], img)

preds = mobilemod(imgTensor)
print(preds.shape)
vis_wrap(preds, img)
boxImg = vis(np.ascontiguousarray(img, dtype=np.uint8), preds[:, 0:4], preds[:, 4]*preds[:, 5],
             torch.ones_like(preds[:, 4]), conf=0.3, class_names=["", "car", "bike"])
plt.imshow(boxImg)
plt.show()
'''
# %% EVAL
imgtransform = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize((416, 416)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.ConvertImageDtype(torch.float32),
        torchvision.transforms.Normalize(
            mean=0,
            std=1/255.0,
        ),
    ]
)


def transforms(img, target):
    w, h = img.width, img.height
    img = imgtransform(img)
    nw, nh = img.shape[1:3]
    hr = nh / h
    wr = nw / w
    for i in range(len(target)):
        target[i]["bbox"][0] = (target[i]["bbox"][0]) * wr
        target[i]["bbox"][1] = (target[i]["bbox"][1]) * hr
        target[i]["bbox"][2] = (target[i]["bbox"][2]) * wr
        target[i]["bbox"][3] = (target[i]["bbox"][3]) * hr
    return img, target


dataset = CocoDetection(
    root="/home/augo/data/COCO/val2017/",
    annFile="/home/augo/data/COCO/annotations/instances_val2017.json",
    transforms=transforms
)
catLookup = dict([(c['id'], i) for (i, c) in enumerate(
    dataset.coco.loadCats(dataset.coco.getCatIds()))])
# %%
# make dataloader
dataloader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=True,
    collate_fn=lambda batch: tuple(zip(*batch)),
)
# %%
inTensor, _ = transforms(pilimg, [])
img = np.copy(np.array(inTensor.permute(1, 2, 0).int()))
# plt.imshow(img)
# plt.show()
out = jitmod(inTensor[None])
print(out.shape, img.shape)
#vis_wrap(out, img, confthresh=0.3)

# %%
def calcMetrics(ious, preds, targets):
    # true prediciton, false prediction, void prediction, missing prediction
    tp, fp, vp, mp = 0, 0, 0, 0
    # iterate over all predictions
    for i in range(len(preds)):
        # check if there is a iou target for this prediction
        if ious[i].any():
            hitidx = ious[i].argmax()
            # check if prediction is correct
            if targets[hitidx] == preds[i]:
                tp += 1
            else:
                fp += 1
            # mark target as used
            targets[hitidx] = -1
            preds[i] = -1
    vp = len(preds[preds != -1])
    mp = len(targets[targets != -1])
    return tp, fp, vp, mp

def matchDetections(boxes, tboxes, targets, iouthresh=0.5):
    # calc ious
    ious = box_iou(boxes, tboxes)
    boxious[boxious < iouthresh] = 0
    labels = np.ones(len(boxes), dtype=np.int32)*-1
    # iterate over all predictions
    for i in range(len(boxes)):
        # check if there is a iou target for this prediction
        if ious[i].any():
            # get best targetidx
            hitidx = ious[i].argmax()
            # check if target is still available
            if targets[hitidx] >= 0:
                # save best target as label
                labels[i] = targets[hitidx]
                # TODO mark target as used??
                #targets[hitidx] = -1
    return labels

# %%
conftresh = 0.3
iouthresh = 0.5
timing = False
timestamp = time()
mlx, mly = [], []
precs, tps, fps, vps, mps = [], [], [], [], []
with torch.no_grad():
    for i, (inputs, targets) in enumerate(dataloader):
        inTensor = torch.stack(inputs)
        img = inputs[0].permute(1, 2, 0)
        if timing:
            print('batch', i, 'prepare inputs', time()-timestamp)
            timestamp = time()
        out = jitmod(inTensor).detach()
        if timing:
            print('batch', i, 'forward pass', time()-timestamp)
            timestamp = time()
        out = out[out[:, 4] >= conftresh]
        pclasses = out[:, 6].int()
        # plt.imshow(img)
        # plt.show()
        try:
            tboxes = torch.Tensor([getBoxCornersFromCOCOBox(t['bbox']) for t in targets[0]])
            tconf = torch.Tensor([[1] for _ in targets[0]])
            tclasses = torch.Tensor([catLookup[t['category_id']] for t in targets[0]]).int()
            tout = torch.cat([tboxes, tconf, tconf, tclasses[:, None]], dim=1)
            if timing:
                print('batch', i, 'formating targets', time()-timestamp)
                timestamp = time()

            # collect ml train data
            mlx.append(out.cpu().numpy())
            mly.append(tout.cpu().numpy())
            #vis_wrap(out, img, confthresh=0.3)
            #vis_wrap(tout, img, confthresh=conftresh, showscore=False)

            boxious = box_iou(out[:, 0:4], tout[:, 0:4])
            boxious[boxious < iouthresh] = 0
            if timing:
                print('batch', i, 'calc IoUs', time()-timestamp)
                timestamp = time()


            tp, fp, vp, mp = calcMetrics(boxious, pclasses, tclasses)
            if timing:
                print('batch', i, 'calc metrics', time()-timestamp)
                timestamp = time()
            precs.append(tp/(tp+fp+vp+mp))
            tps.append(tp)
            fps.append(fp)
            vps.append(vp)
            mps.append(mp)
            if i and not i % 50:
                base = sum(tps+fps+vps+mps)
                print(i, sum(tps), sum(fps), sum(vps), sum(mps),'avg prec:', round(sum(precs)/len(precs), 3), 'avg tp:', round(sum(tps)/base, 3), 'avg fp:',
                        round(sum(fps)/base, 3), 'avg vp:', round(sum(vps)/base, 3), 'avg mp:', round(sum(mps)/base), end='\r')
        except IndexError as e:
            print(e, [t['bbox'] for t in targets[0]])
            continue
np.save('mlx.npy', np.array(mlx, dtype=object))
np.save('mly.npy', np.array(mly, dtype=object))
#%% LOAD ML DATA
mlx = np.load('mlx.npy', allow_pickle=True)
mly = np.load('mly.npy', allow_pickle=True)
cleanmask = [e.shape[0]>0 for e in mly]
mlx = [mlx[i] for i in range(len(mlx)) if cleanmask[i]]
mly = [mly[i] for i in range(len(mly)) if cleanmask[i]]
#%% TRAIN ML
knn = neighbors.KNeighborsClassifier(n_neighbors=5)
train_percent = 0.05
# get labels
boxes = [e[:, 0:4] for e in mlx]
tboxes = [e[:, 0:4] for e in mly]
targets = [(e[:, 6]).astype(np.int32) for e in mly]
# match detections
labels = [matchDetections(torch.from_numpy(b), torch.from_numpy(tb), t, iouthresh=0.5) for b, tb, t in zip(boxes, tboxes, targets)]
# flatten
features = np.concatenate(mlx, axis=0)[:,7:]
labels = np.concatenate(labels, axis=0)
# split
train_size = int(len(features) * train_percent)
train_features = features[:train_size]
train_labels = labels[:train_size]
test_features = features[train_size:]
test_labels = labels[train_size:]
print(int(len(boxes)* train_percent), "train images for all 80 classes with",
    len(train_features), "detections =>\n", len(train_features)/80, "detections per class")
# train
knn.fit(train_features, train_labels)
# test
print('score', knn.score(test_features, test_labels))
# %% Seems like batching is not speeding it up
nruns = 10
inTensor = torch.rand(8, 3, 416, 416)
#print(f'standard model {nruns} runs', timeit(lambda: model(inTensor), number=nruns))
#print(f'jitmodel {nruns} runs', timeit(lambda: jitmod(inTensor), number=nruns))
with torch.no_grad():
    print(f'jitmodel no grad {nruns} runs', timeit(lambda: jitmod(inTensor), number=nruns))
# slow: print(f'mobile model {nruns} runs', timeit(lambda: mobilemod(inTensor), number=nruns))
# %%

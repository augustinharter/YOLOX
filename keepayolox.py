# %%
from copy import copy, deepcopy
import sklearn.neighbors as neighbors
from pycocotools.coco import COCO
from time import time
from timeit import timeit
from torchvision.ops import box_iou
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from yolox.utils.boxes import postprocess
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
CLASS_NAMES = []

# %%
def getBoxCornersFromCOCOBox(box):
    return (box[0], box[1], box[0]+box[2], box[1]+box[3])


def custompostprocesssingle(prediction, num_classes, conf_thre=0.7, nms_thre=0.45, class_agnostic=False, classconfscore=False):
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

    conf_mask = (image_pred[:, 4] * (class_conf.squeeze() if classconfscore else 1)>= conf_thre).squeeze()
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
#ckpt_file = f'{"yolox_"+modeltype}.pth'
ckpt_file = f'{"yolox_"+modeltype+"_rpc"}.pth'
ckpt_file = f'{"yolox_"+modeltype}.pth'
ckpt_file = 'YOLOX_outputs/yolox_tiny_rpc/best_ckpt.pth'
exp_file = None
exp_file = f'{"exps/example/custom/yolox_"+modeltype+"_rpc"}.py'
name = "yolox-"+modeltype
'''
# demo
!python tools/demo.py image - n {name} - c {ckpt_file} - -save_result \
    - -path / home/augo/coding/keepa/ml/automl/efficientdet/carsandbikes416.png
!python tools/export_torchscript.py - -output-name {outputname} - n {name} - c {ckpt_file}\
    - -decode_in_inference

python tools/train.py -c yolox_tiny_rpc.pth -n yolox-tiny -cpu -f exps/example/custom/yolox_tiny_rpc.py -b 12
'''
# %% LOAD NORMAL MODEL
# load the model state dict
exp = get_exp(exp_file, name)
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

        out = custompostprocesssingle(out, exp.num_classes, exp.test_conf, exp.nmsthre, 
                                    class_agnostic= True, classconfscore=True)
        #out = postprocess(
        #            out[None], exp.num_classes, exp.test_conf, exp.nmsthre, class_agnostic=True
        #        )[0]
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
# load mobile model
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
    print(out.shape)
    boxImg = vis(np.ascontiguousarray(img, dtype=np.uint8), out[:, 0:4], out[:, 4],  # *out[:,5],
                 out[:, 6], conf=confthresh, class_names=CLASS_NAMES, showscore=showscore)
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

'''
preds = mobilemod(imgTensor)
print(preds.shape)
#vis_wrap(preds, img)
#boxImg = vis(np.ascontiguousarray(img, dtype=np.uint8), preds[:, 0:4], preds[:, 4]*preds[:, 5],
#             torch.ones_like(preds[:, 4]), conf=0.3, class_names=["", "car", "bike"])
#plt.imshow(boxImg)
#plt.show()
# %% EVAL
imgtransform = torchvision.transforms.Compose(
    [
        #torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize((416, 416), interpolation=Image.NEAREST),
        torchvision.transforms.PILToTensor(),
        lambda x: x.float(),
        #lambda x: (x.float(),print(x.float()))[0],
        #torchvision.transforms.Normalize(
        #    mean=0,
        #    std=1/255.0,
        #),
    ]
)

def transforms(img, target):
    w, h = img.width, img.height
    img = imgtransform(img)
    nw, nh = img.shape[1:3]
    hr = nh / h
    wr = nw / w
    target = deepcopy(target)
    for i in range(len(target)):
        target[i]["bbox"][0] = (target[i]["bbox"][0]) * wr
        target[i]["bbox"][1] = (target[i]["bbox"][1]) * hr
        target[i]["bbox"][2] = (target[i]["bbox"][2]) * wr
        target[i]["bbox"][3] = (target[i]["bbox"][3]) * hr
    return img, target
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

def matchDetectionsWrong(boxes, tboxes, target_labels, iouthresh=0.5, deplete_tlabels=True):
    # calc ious
    ious = box_iou(boxes, tboxes)
    ious[ious < iouthresh] = 0
    print(ious.shape)
    # initialize labels as VOID (-1)
    labels = torch.ones(len(boxes))*-1
    # iterate over all predictions
    for i in range(len(boxes)):
        # check if there is a iou target for this prediction
        if ious[i].any():
            # get hit = best targetidx
            hitidx = ious[i].argmax()
            # check if target is still available
            if target_labels[hitidx] >= 0:
                # save best target as label
                labels[i] = int(target_labels[hitidx])
                if deplete_tlabels:
                    # mark target as used so it will become a -1 label for next match
                    target_labels[hitidx] = -1
    return labels, target_labels

def matchDetections(boxes, tboxes, tlabels, iouthresh=0.5, deplete_tlabels=True):
    # calc ious
    ious = box_iou(tboxes, boxes)
    ious[ious < iouthresh] = 0
    #print(ious.shape)
    # initialize labels as VOID (-1)
    plabels = torch.ones(len(boxes))*-1
    # iterate over all targets
    for i in range(len(tboxes)):
        # check if there is a iou target for this prediction
        if ious[i].any():
            # get hit = best prediciton idx
            hitidx = ious[i].argmax()
            # save label for best prediction
            plabels[hitidx] = int(tlabels[i])
            if deplete_tlabels:
                # mark target as used
                tlabels[i] = -1
    return plabels, tlabels

#%%
#splitname = 'test2019'
datasetname = 'COCO'
splitname = 'val2017'
datasetname = 'retail_product_checkout'
splitname = 'val2019'
dataset = CocoDetection(
    root=f"/home/augo/data/{datasetname}/{splitname}/",
    annFile=f"/home/augo/data/{datasetname}/annotations/instances_{splitname}.json",
    transforms=transforms
)
catsInfo = dataset.coco.loadCats(dataset.coco.getCatIds())
catLookup = dict([(c['id'], i) for (i, c) in enumerate(
    catsInfo)])
CLASS_NAMES = [c['name'] for (i, c) in enumerate(catsInfo)]
# %%
# make dataloader
dataloader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    collate_fn=lambda batch: tuple(zip(*batch)),
)
'''
inTensor, _ = transforms(pilimg, [])
img = np.copy(np.array(inTensor.permute(1, 2, 0).int()))
# plt.imshow(img)
# plt.show()
out = jitmod(inTensor[None])
print(out.shape, img.shape)
#vis_wrap(out, img, confthresh=0.3)
'''
print('dataset', len(dataset), 'dataloader', len(dataloader))

#%%
knn = None
#%%
feature_limit = 7+(96*2)
conftresh = 0.3
iouthresh = 0.5
timing = False
timestamp = time()
rawx, rawy = [], []
metrics = []
with torch.no_grad():
    for i, (inputs, targets) in enumerate(dataloader):
        inTensor = torch.stack(inputs).flip(1)
        img = inTensor[0].permute(1, 2, 0)
        #print(inTensor[0].flatten(), torch.std_mean(inTensor[0].flatten()), inTensor[0].max(), inTensor[0].min())

        if timing:
            print('batch', i, 'prepare inputs', time()-timestamp)
            timestamp = time()
        #print("eval stats", inTensor.shape, inTensor.dtype, inTensor.max(), inTensor.min())
        out = jitmod(inTensor).detach()
        if timing:
            print('batch', i, 'forward pass', time()-timestamp)
            timestamp = time()
        out = out[out[:, 4] >= conftresh]
        pclasses = out[:, 6].int()

        # KNN
        if out.shape[0] and knn:
            infeatures = out[:, 7:feature_limit].cpu().numpy()
            #idxs = [((torch.from_numpy(train_features) - af).sum(1)==0).int().argmax().item() for af in infeatures]
            pclasses = torch.from_numpy(knn.predict(infeatures)).int()
            #print(pclasses)

        # plt.imshow(img)
        # plt.show()
        tboxes = torch.Tensor([getBoxCornersFromCOCOBox(t['bbox']) for t in targets[0]])
        tconf = torch.Tensor([[1] for _ in targets[0]])
        tclasses = torch.Tensor([catLookup[t['category_id']] for t in targets[0]]).int()
        tout = torch.cat([tboxes, tconf, tconf, tclasses[:, None]], dim=1)
        if timing:
            print('batch', i, 'formating targets', time()-timestamp)
            timestamp = time()
        
        if not (out.shape[0] and tout.shape[0]):
            continue
        # collect ml train data
        rawx.append(out.cpu().numpy())
        rawy.append(tout.cpu().numpy())
        if targets[0][0]['image_id'] == 220:
            vis_wrap(out, img, confthresh=conftresh)
            vis_wrap(tout, img, confthresh=conftresh, showscore=False)
        
        #print(tclasses.numpy(), '|', pclasses.numpy())
        # MATCH AND EVAL PREDICTIONS
        matchedTargets, allTargets = matchDetections(out[:, :4].clone(), tboxes.clone(), tclasses.clone(), 
            iouthresh=iouthresh, deplete_tlabels=True)
        tp = (matchedTargets == pclasses).sum().item()
        tvp = (matchedTargets[matchedTargets == pclasses] == -1).sum().item()
        tp = tp - tvp
        remainTargets = matchedTargets[matchedTargets != pclasses]
        vp = (remainTargets == -1).sum().item()
        fp = (remainTargets != -1).sum().item()
        mp = (allTargets != -1).sum().item()
        #print(f'batch {i}: {tp} tp, {fp} fp, {mp} mp, {tvp} tvp, {vp} vp, {tp+mp+fp} total, {len(tclasses)} targets')

        if timing:
            print('batch', i, 'calc metrics', time()-timestamp)
            timestamp = time()
        metrics.append(np.array((tp, fp, mp, tvp, vp, len(tclasses))))
        if i and not i % 50:
            #print(idxs,'\n labels', train_labels[idxs],'\n predicted', pclasses.numpy())
            print('matched', matchedTargets.numpy())
            print('all targets', allTargets.numpy())
            summed = np.array(metrics).sum(axis=0)
            #base = summed[[0,1,2]].sum()
            base = summed[[0,1,2,3,4]].sum()
            #print(f'batch {i}: {(summed[0]/base).round(3)} tp, {(summed[1]/base).round(3)} fp, {(summed[2]/base).round(3)} mp, {(summed[3]/base).round(3)} tvp, {(summed[4]/base).round(3)} vp, {base} total, {summed[5]} targets')
            #print(f'batch {i}: {((summed[0]+summed[3])/base).round(3)} correct = tp+tvp, {((summed[1]+summed[2]+summed[4])/base).round(3)} wrong = mp+vp, {(summed[2]/base).round(3)} mp, {(summed[3]/base).round(3)} tvp, {(summed[4]/base).round(3)} vp, {base} total, {summed[5]} targets')
            print(f'b{i} knn={knn}:  \
                \n tp {summed[0]}, fp {summed[1]}, mp {summed[2]}, tvp {summed[3]}, vp {summed[4]}, total {base} targets {summed[5]} \
                \n prec = {round((summed[0]+summed[3])/(summed[0]+summed[3]+summed[4]+summed[1]), 3)} = (tp+tvp) ÷ (tp+tvp+vp+fp) \
                \n recall = {round((summed[0])/(summed[0]+summed[1]+summed[2]), 3)} = (tp) ÷ (tp+fp+mp) \
                \n obj-prec {round( (summed[0]+summed[1])/(summed[0]+summed[1]+summed[2]), 3)}')

np.save(f'{splitname}-rawx.npy', np.array(rawx, dtype=object))
np.save(f'{splitname}-rawy.npy', np.array(rawy, dtype=object))
print(f"saved raw data with name {splitname}-rawx.npy and {splitname}-rawy.npy")
#%% LOAD RAW ML DATA
splitname = 'val2019'
rawx = np.load(f'{splitname}-rawx.npy', allow_pickle=True)
rawy = np.load(f'{splitname}-rawy.npy', allow_pickle=True)
print("loaded raw data")
cleanmask = [e.shape[0]>0 for e in rawy]
rawx = [rawx[i] for i in range(len(rawx)) if cleanmask[i]]
rawy = [rawy[i] for i in range(len(rawy)) if cleanmask[i]]
print(sum(cleanmask), 'clean samples out of', len(cleanmask))

#%% PROCESS RAW ML DATA
# get labels
boxes = [e[:, 0:4] for e in rawx]
tboxes = [e[:, 0:4] for e in rawy]
targets = [(e[:, 6]).astype(np.int32) for e in rawy]
# match detections
labels = [matchDetections(torch.from_numpy(b), torch.from_numpy(tb), 
    torch.from_numpy(t), iouthresh=iouthresh, deplete_tlabels=True)[0]
    for b, tb, t in zip(boxes, tboxes, targets)]
# flatten
features = np.concatenate(rawx, axis=0)[:,7:feature_limit]
labels = np.concatenate(labels, axis=0)
print('done processing', features.shape)

#%% SAVE ML DATA
np.save(f'{splitname}-features.npy', features)
np.save(f'{splitname}-labels.npy', labels)

#%% LOAD ML DATA
splitname = 'val2019'
features = np.load(f'{splitname}-features.npy')
labels = np.load(f'{splitname}-labels.npy')
print('loaded features', features.shape)

#%% EVAL KNN
splits = [0.01, 0.02, 0.05, 0.1]
nneighbs = [1, 3, 5]
scores = []
for nneigh in nneighbs:
    for train_percent in splits:
        # shuffle features array
        order = np.arange(len(features))
        np.random.shuffle(order)
        features = features[order]
        labels = labels[order]
        train_size = int(len(features) * train_percent)
        train_features = features[:train_size]
        train_labels = labels[:train_size]
        test_features = features[train_size:]
        test_labels = labels[train_size:]
        print(int(len(rawx)* train_percent), "train images for all 80 classes with",
            len(train_labels), "detections =>\n", len(train_labels)/80, "detections per class")
        print(int(len(rawx)* (1-train_percent)), "test images for all 80 classes with",
            len(test_labels), "detections =>\n", len(test_labels)/80, "detections per class")

        knn = neighbors.KNeighborsClassifier(n_neighbors=nneigh)#, weights='distance')
        knn.fit(train_features, train_labels)

        print('train score', knn.score(train_features, train_labels))
        testscore = knn.score(test_features, test_labels)
        print('test score', testscore)
        scores.append(testscore)
knn_scores = np.split(np.array(scores), len(nneighbs))
for i,score in enumerate(knn_scores):
    plt.scatter([f'{s}' for s in splits], score, label=f'knn={nneighbs[i]}')
plt.xlabel('train percent')
plt.ylabel('test score')
plt.ylim(0,1)
plt.title(f' KNN score vs. train percent')
plt.legend(loc='lower right')
plt.savefig(f'knn_score_vs_train_percent_{splitname}.png')
plt.show()
# %% Seems like batching is not speeding it up
nruns = 10
inTensor = torch.rand(8, 3, 416, 416)
#print(f'standard model {nruns} runs', timeit(lambda: model(inTensor), number=nruns))
#print(f'jitmodel {nruns} runs', timeit(lambda: jitmod(inTensor), number=nruns))
with torch.no_grad():
    print(f'jitmodel no grad {nruns} runs', timeit(lambda: jitmod(inTensor), number=nruns))
# slow: print(f'mobile model {nruns} runs', timeit(lambda: mobilemod(inTensor), number=nruns))
# %%

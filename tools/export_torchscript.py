#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os
from turtle import forward
from loguru import logger
from torch.utils.mobile_optimizer import optimize_for_mobile
import torchvision

import torch

# export cwd as python path
import sys
sys.path.append(os.getcwd())
from yolox.utils.model_utils import get_model_info
from yolox.exp import get_exp


def make_parser():
    parser = argparse.ArgumentParser("YOLOX torchscript deploy")
    parser.add_argument(
        "--output-name", type=str, default="yolox.torchscript.pt", help="output name of models"
    )
    parser.add_argument("--batch-size", type=int, default=1, help="batch size")
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="experiment description file",
    )
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt path")
    parser.add_argument(
        "--decode_in_inference",
        action="store_true",
        help="decode in inference or not"
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--includepost",
        action="store_true",
        help="postprocess in inference or not"
    )

    return parser


@logger.catch
def main():
    args = make_parser().parse_args()
    logger.info("args value: {}".format(args))
    exp = get_exp(args.exp_file, args.name)
    exp.merge(args.opts)

    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    model = exp.get_model()
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))
    if args.ckpt is None:
        file_name = os.path.join(exp.output_dir, args.experiment_name)
        ckpt_file = os.path.join(file_name, "best_ckpt.pth")
    else:
        ckpt_file = args.ckpt

    # load the model state dict
    print('ckpt file', ckpt_file)
    ckpt = torch.load(ckpt_file, map_location="cpu")

    model.eval()
    if "model" in ckpt:
        ckpt = ckpt["model"]
    model.load_state_dict(ckpt)
    model.head.decode_in_inference = args.decode_in_inference

    #modelwrap = lambda x: custompostprocess(model(x).detach(), exp.num_classes, exp.test_conf, exp.nmsthre, True)
    def modelwrap(x):
        out = model(x)
        nms_out_index = torchvision.ops.nms(
                    out[:, :4],
                    out[:, 4],
                    0.45,
                )
        out = out[nms_out_index]
        return out
    modelwrap = lambda x : model(x)

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
            out = self.model(x)
            out = custompostprocess(out, exp.num_classes, exp.test_conf, exp.nmsthre, True)
            return out

    print('NUM CLASSES, TEST CONF, NMSTHRESH',exp.num_classes, exp.test_conf, exp.nmsthre)
    wrapper = Wrapper(model)

    logger.info("loading checkpoint done.")
    dummy_input = torch.randn(args.batch_size, 3, exp.test_size[0], exp.test_size[1]).detach()

    mod = torch.jit.trace(model if not args.includepost else wrapper, dummy_input)
    #mod.save(args.output_name)
    #logger.info("generated torchscript model named {}".format(args.output_name))

    optimized_traced_model = optimize_for_mobile(mod)
    outpath = f"/home/augo/coding/keepa/app/keepa_auth/assets/models/{args.output_name}"
    optimized_traced_model._save_for_lite_interpreter(outpath)
    logger.info("generated torchscript mobile model at {}".format(outpath))

    

def custompostprocess(prediction, num_classes, conf_thre=0.7, nms_thre=0.45, class_agnostic=False):
    print('out stats:', 
        prediction[0].shape, '\n max:', 
        prediction[0].max(), '\n min:', 
        prediction[0].min(), '\n std classes :',
        prediction[0][:,5:].flatten().std(), '\n mean:',
        prediction[0][:,5:].flatten().mean(), '\n boxes:',
        prediction[0][:,:4].flatten().std(),
        prediction[0][:,:4].flatten().mean(),
    )
    box_corner = prediction.new(prediction.shape).detach()
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    for i, image_pred in enumerate(prediction):

        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Get score and class with highest confidence
        class_conf, class_pred = torch.max(image_pred[:, 5: 5 + num_classes], 1, keepdim=True)

        conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thre).squeeze()
        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)
        detections = detections[conf_mask]
        if not detections.size(0):
            continue

        if class_agnostic:
            nms_out_index = torchvision.ops.nms(
                detections[:, :4],
                detections[:, 4] * detections[:, 5],
                nms_thre,
            )
        else:
            nms_out_index = torchvision.ops.batched_nms(
                detections[:, :4],
                detections[:, 4] * detections[:, 5],
                detections[:, 6],
                nms_thre,
            )

        detections = detections[nms_out_index]
        if output[i] is None:
            output[i] = detections
        else:
            output[i] = torch.cat((output[i], detections))

    return tuple(output)

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

    conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thre).squeeze()
    # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
    detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)
    detections = detections[conf_mask]

    if class_agnostic:
        nms_out_index = torchvision.ops.nms(
            detections[:, :4],
            detections[:, 4] * detections[:, 5],
            nms_thre,
        )
    else:
        nms_out_index = torchvision.ops.batched_nms(
            detections[:, :4],
            detections[:, 4] * detections[:, 5],
            detections[:, 6],
            nms_thre,
        )

    detections = detections[nms_out_index]
 

    return detections
if __name__ == "__main__":
    main()

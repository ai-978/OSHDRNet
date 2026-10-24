# -*- coding:utf-8 -*-
"""Evaluation of an OSHDRNet checkpoint on a test set (reflect padding to a multiple of 128, as in AFUNet).
    python test.py [--pretrained_model logdir_OSHDRNet_ODC/ckpt/final.pth] [--save_results]

Reports PSNR / SSIM in the linear and in the mu-law domain of the final output and of O_det (per scene and
on average), computed as in the test.py with which AFUNet reports its results (see scene_metrics).  With
--save_results, save_dir receives for every scene the HDR result (.hdr), a mu-law tone-mapped preview (.png),
the completion region M_comp (_mcomp.png) and, for the whole set, metrics.json.
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio
from torch.utils.data import DataLoader

from dataset.test_dataset import Test_Dataset
from models.OSHDRNet import OSHDRNet, model_options, mu_expand
from utils.utils import (AverageMeter, calculate_ssim, mechanism_stats, pad_for_test, range_compressor, save_hdr,
                         save_mask, save_tonemapped)

parser = argparse.ArgumentParser(description="OSHDRNet test", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--pretrained_model", "--checkpoint", dest="checkpoint", type=str,
                    default="logdir_OSHDRNet_ODC/ckpt/final.pth",
                    help="model of train_odc.py (final.pth: last epoch) or of train.py with --no_odc")
parser.add_argument("--dataset_dir", type=str, default="../data/Kal", help="dataset directory")
parser.add_argument("--test_path", "--split", dest="test_path", type=str, default="Test",
                    help="test scenes in dataset_dir")
parser.add_argument("--exposure_file_name", type=str, default="exposure.txt",
                    help="exposure file name (Hu's dataset: input_exp.txt)")
parser.add_argument("--label_file_name", type=str, default="HDRImg.hdr",
                    help="label file name (Hu's dataset: ref_hdr_aligned_linear.hdr)")
parser.add_argument("--patch_size", type=int, default=128,
                    help="reflect padding to a multiple of this size (AFUNet: 128; 1000 x 1500 -> 1024 x 1536)")
parser.add_argument("--num_workers", type=int, default=0, help="workers to fetch data (scenes are cached)")
parser.add_argument("--no_odc", action="store_true", default=False, help="evaluate O_det only")
parser.add_argument("--save_results", action="store_true", default=False,
                    help="save HDR results, tone-mapped previews, M_comp and metrics.json")
parser.add_argument("--save_dir", type=str, default="./results/", help="output directory of --save_results")
parser.add_argument("--no_cuda", action="store_true", default=False, help="disables CUDA")


def scene_metrics(pred_img, label):
    """PSNR / SSIM of one scene as reported by AFUNet's test.py.  pred_img, label: 3 x H x W float32 linear HDR.
    PSNR: skimage with data_range 1; SSIM: calculate_ssim (11 x 11 Gaussian window, valid region without the
    5-pixel border) of the images scaled by 255 and clipped to [0, 255]; mu-law domain: range_compressor."""
    # psnr-l and psnr-mu
    scene_psnr_l = peak_signal_noise_ratio(label, pred_img, data_range=1.0)
    label_mu = range_compressor(label)
    pred_img_mu = range_compressor(pred_img)
    scene_psnr_mu = peak_signal_noise_ratio(label_mu, pred_img_mu, data_range=1.0)
    # ssim-l
    scene_ssim_l = calculate_ssim(np.clip(pred_img * 255.0, 0., 255.).transpose(1, 2, 0),
                                  np.clip(label * 255.0, 0., 255.).transpose(1, 2, 0))
    # ssim-mu
    scene_ssim_mu = calculate_ssim(np.clip(pred_img_mu * 255.0, 0., 255.).transpose(1, 2, 0),
                                   np.clip(label_mu * 255.0, 0., 255.).transpose(1, 2, 0))
    return {"psnr_l": float(scene_psnr_l), "psnr_mu": float(scene_psnr_mu),
            "ssim_l": float(scene_ssim_l), "ssim_mu": float(scene_ssim_mu)}


def main():
    # Settings
    args = parser.parse_args()
    print(">>>>>>>>> Start Testing >>>>>>>>>")
    print("Load weights from: ", args.checkpoint)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Device: ", device)

    # model (options stored with the checkpoint)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = OSHDRNet(**model_options(checkpoint.get("config", {}).get("model", {})))
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(">>>>>>> Model Loaded: epoch {}, {:,} parameters >>>>>>>>>".format(checkpoint.get("epoch", "-"), n_params))

    # data
    test_dataset = Test_Dataset(args.dataset_dir, args.test_path, args.exposure_file_name, args.label_file_name)
    loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
    print("testset_size: ", len(test_dataset))
    if args.save_results:
        os.makedirs(args.save_dir, exist_ok=True)

    meters, stats_total, per_scene, seconds = {}, {}, [], []
    with torch.no_grad():
        for idx, batch in enumerate(loader):
            inputs = torch.cat([t.to(device) for t in batch["inputs"]], dim=1)
            target = batch["label"].to(device)
            h, w = target.shape[-2:]
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            out = model.forward_features(pad_for_test(inputs, args.patch_size), use_odc=not args.no_odc)
            if device.type == "cuda":
                torch.cuda.synchronize()
            seconds.append(time.time() - t0)

            # psnr / ssim in the linear and in the mu-law domain (as reported by AFUNet)
            name = batch["name"][0]
            scene = {"name": name}
            label = target[0].cpu().numpy().astype(np.float32)
            for prefix, key in (("det", "O_det_mu"), ("final", "O_mu")):
                pred_img = mu_expand(out[key][0, :, :h, :w]).cpu().numpy().astype(np.float32)
                for metric, value in scene_metrics(pred_img, label).items():
                    scene[prefix + "_" + metric] = value
                    meters.setdefault(prefix + "_" + metric, AverageMeter()).update(value)
            for metric, value in mechanism_stats(out).items():
                stats_total[metric] = stats_total.get(metric, 0.0) + value
            per_scene.append(scene)
            print(" {} {} | PSNR_mu: {:.4f}  PSNR_l: {:.4f} | SSIM_mu: {:.4f}  SSIM_l: {:.4f}".format(
                idx, name, scene["final_psnr_mu"], scene["final_psnr_l"], scene["final_ssim_mu"],
                scene["final_ssim_l"]))

            if args.save_results:
                image_mu = out["O_mu"][0, :, :h, :w].permute(1, 2, 0).cpu().numpy()
                image_hdr = mu_expand(out["O_mu"][0, :, :h, :w]).permute(1, 2, 0).cpu().numpy()
                save_hdr(os.path.join(args.save_dir, name + ".hdr"), np.ascontiguousarray(image_hdr, dtype=np.float32))
                save_tonemapped(os.path.join(args.save_dir, name + ".png"), image_mu)
                if "M_comp" in out:
                    save_mask(os.path.join(args.save_dir, name + "_mcomp.png"), out["M_comp"][0, 0].cpu().numpy())

    n = len(per_scene)
    result = {"checkpoint": args.checkpoint, "metrics": {k: meter.avg for k, meter in meters.items()},
              "mechanism": {k: v / n for k, v in stats_total.items()},
              "seconds_per_image": float(np.mean(seconds[1:] if n > 1 else seconds)),
              "parameters": n_params, "per_scene": per_scene}
    print("Average PSNR_mu: {:.4f}  PSNR_l: {:.4f}".format(result["metrics"]["final_psnr_mu"],
                                                           result["metrics"]["final_psnr_l"]))
    print("Average SSIM_mu: {:.4f}  SSIM_l: {:.4f}".format(result["metrics"]["final_ssim_mu"],
                                                           result["metrics"]["final_ssim_l"]))
    print(json.dumps({k: v for k, v in result.items() if k != "per_scene"}, indent=1))
    if args.save_results:
        with open(os.path.join(args.save_dir, "metrics.json"), "w") as f:
            json.dump(result, f, indent=1)
    print(">>>>>>>>> Finish Testing >>>>>>>>>")


if __name__ == "__main__":
    main()

# -*- coding:utf-8 -*-
"""Training of OSHDRNet (paper Sec. III-F).

python train.py       stage 1: RGA and DEA jointly, L_det = ||O_det^mu - Y^mu||_1 + lambda_fft ||F(O_det^mu - Y^mu)||_1
python train_odc.py   stage 2: RGA and DEA frozen; C_theta on the completion regions they produce,
                      L_comp = ||M_comp (O_comp^mu - Y^mu)||_1 / (||M_comp||_1 + eps)
Adam, cosine decay of the learning rate to --eta_min; random crops with flipping.

The model of the last epoch is the result of a stage (ckpt/final.pth); the test set is evaluated during
training for monitoring only and is not used to select a checkpoint.
Outputs in --logdir: ckpt/final.pth, ckpt/epoch_<N>.pth (every --save_every epochs), ckpt/latest.pth (to
resume), config.json, train.log (losses, test metrics and mechanism statistics, one JSON record per epoch),
test.log (test metrics) and tensorboard/ (curves).
"""
import argparse
import json
import os
import time
from pathlib import Path

# full-resolution evaluation after training steps fragments the allocator otherwise (OOM on a 24 GB GPU)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset.test_dataset import Test_Dataset
from dataset.training_dataset import Training_Dataset
from loss.loss import CompletionLoss, DetectionLoss
from lr_scheduler.mylr import cosine_lr
from models.OSHDRNet import OSHDRNet, model_options, mu_compress, mu_expand, set_compile
from utils.utils import AverageMeter, get_logger, hdr_metrics, mechanism_stats, pad_for_test, set_random_seed

# settings that differ between the two training stages (the hyper-parameters of the paper have no defaults)
STAGES = {
    "det": {"description": "OSHDRNet training, stage 1: RGA and DEA",
            "logdir": "logdir_OSHDRNet", "eval_every": 5, "save_every": 5},
    "comp": {"description": "OSHDRNet training, stage 2: ODC (RGA and DEA frozen)",
             "logdir": "logdir_OSHDRNet_ODC", "eval_every": 1, "save_every": 1},
}


class HelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    """Shows the defaults of the optional settings; the hyper-parameters of the paper are required."""

    def _get_help_string(self, action):
        return action.help if action.required else super()._get_help_string(action)


def get_args(stage):
    defaults = STAGES[stage]
    parser = argparse.ArgumentParser(description=defaults["description"], formatter_class=HelpFormatter)
    # Directories
    parser.add_argument("--logdir", "--output_dir", dest="output_dir", type=str, default=defaults["logdir"],
                        help="target log / checkpoint directory")
    if stage == "comp":
        parser.add_argument("--init", type=str,
                            default=os.path.join(STAGES["det"]["logdir"], "ckpt", "final.pth"),
                            help="stage-1 model (last epoch) whose RGA and DEA are frozen")
    parser.add_argument("--resume", type=str, default="", help="latest.pth of an interrupted run")

    # Dataset settings
    parser.add_argument("--dataset_dir", type=str, default="../data/Kal", help="dataset directory")
    parser.add_argument("--train_path", type=str, default="Training", help="training scenes in dataset_dir")
    parser.add_argument("--test_path", type=str, default="Test", help="test scenes in dataset_dir")
    parser.add_argument("--exposure_file_name", type=str, default="exposure.txt", help="exposure file name")
    parser.add_argument("--label_file_name", type=str, default="HDRImg.hdr", help="label file name")
    parser.add_argument("--patch_size", type=int, required=True, help="patch size for training (paper)")
    parser.add_argument("--test_patch_size", type=int, default=128,
                        help="reflect padding to a multiple of this size at test time (as in AFUNet)")
    parser.add_argument("--repeat", type=int, default=100, help="repetitions of the training scenes per epoch")
    parser.add_argument("--num_workers", type=int, default=6, help="workers to fetch training data")
    parser.add_argument("--test_num_workers", type=int, default=0,
                        help="workers to fetch test data (the test scenes are cached in memory)")

    # Training settings
    parser.add_argument("--epochs", type=int, required=True, help="number of epochs of this stage (paper)")
    parser.add_argument("--batch_size", type=int, required=True, help="training batch size (paper)")
    parser.add_argument("--lr", type=float, required=True, help="initial learning rate (paper)")
    parser.add_argument("--eta_min", type=float, required=True, help="final learning rate of the cosine decay (paper)")
    if stage == "det":
        parser.add_argument("--lambda_fft", type=float, required=True,
                            help="weight of the frequency-domain loss (paper)")
    parser.add_argument("--clip_grad", type=float, default=1.0, help="gradient-norm clipping (0: off)")
    parser.add_argument("--eval_every", type=int, default=defaults["eval_every"],
                        help="test-set evaluation interval in epochs (monitoring only)")
    parser.add_argument("--save_every", type=int, default=defaults["save_every"],
                        help="interval in epochs of the saved checkpoints epoch_<N>.pth")
    parser.add_argument("--log_interval", type=int, default=100, help="steps between training log records")
    parser.add_argument("--seed", type=int, default=923, help="random seed")

    # Speed settings
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True,
                        help="torch.compile the DPA / DEA distribution functions on the GPU (same maths)")

    # Other settings
    parser.add_argument("--no_cuda", action="store_true", default=False, help="disables CUDA training")
    args = parser.parse_args()
    args.stage = stage
    return args


def train_one_epoch(args, model, device, train_loader, optimizer, scheduler, criterion, epoch, log, tb_writer):
    """One epoch of stage 1 (det) or stage 2 (comp); returns the mean losses of the epoch."""
    model.train()
    if args.stage == "comp":                         # RGA and DEA stay frozen, only C_theta is trained
        model.eval()
        model.odc.train()
    use_odc = args.stage == "comp"
    params = [p for group in optimizer.param_groups for p in group["params"]]
    steps_per_epoch = len(train_loader)
    meters, skipped, t0 = {}, 0, time.time()
    for index, batch in enumerate(tqdm(train_loader, ncols=80, desc="epoch %d" % epoch)):
        inputs = torch.cat([t.to(device, non_blocking=True) for t in batch["inputs"]], dim=1)
        target_mu = mu_compress(batch["label"].to(device, non_blocking=True))
        out = model.forward_features(inputs, use_odc=use_odc)

        # loss
        if args.stage == "det":
            loss, parts = criterion(out["O_det_mu"], target_mu)
        else:
            loss, parts = criterion(out["O_comp_mu"], target_mu, out["M_comp"])
        if not torch.isfinite(loss):
            skipped += 1
            optimizer.zero_grad(set_to_none=True)
            continue

        # backward
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(params, args.clip_grad)
        optimizer.step()
        scheduler.step()

        values = {"loss": float(loss.detach())}
        values.update({name: float(value.detach()) for name, value in parts.items()})
        for name, value in values.items():
            meters.setdefault(name, AverageMeter()).update(value)
        if index % args.log_interval == 0:
            stats = mechanism_stats(out)
            brief = {k: round(v, 4) for k, v in stats.items() if k[:2] in ("r_", "m_", "et", "e_", "M_", "al")
                     and (k.endswith(("1", "4")) or k in ("e_r", "M_comp"))}
            log.write("epoch=%d step=%d/%d loss=%.6f lr=%.2e %s %.1fs\n" % (
                epoch, index, steps_per_epoch, float(loss), scheduler.get_last_lr()[0], json.dumps(brief),
                time.time() - t0))
            global_step = (epoch - 1) * steps_per_epoch + index
            for name, value in values.items():
                tb_writer.add_scalar("train/" + name, value, global_step)
            for name, value in brief.items():
                tb_writer.add_scalar("mechanism/" + name, value, global_step)
    train_stats = {name: meter.avg for name, meter in meters.items()}
    train_stats.update({"skipped": skipped, "seconds": time.time() - t0})
    return train_stats


@torch.no_grad()
def evaluate(model, loader, device, multiple, use_odc):
    """Test-set metrics of O_det (and of the final output with ODC) and mechanism statistics."""
    model.eval()
    totals, stats_total, n = {}, {}, 0
    for batch in tqdm(loader, ncols=80, desc="test"):
        inputs = torch.cat([t.to(device) for t in batch["inputs"]], dim=1)
        target = batch["label"].to(device)
        h, w = target.shape[-2:]
        out = model.forward_features(pad_for_test(inputs, multiple), use_odc=use_odc)
        results = {"det": out["O_det_mu"]}
        if use_odc:
            results["final"] = out["O_mu"]
        for prefix, prediction_mu in results.items():
            for name, value in hdr_metrics(mu_expand(prediction_mu[..., :h, :w]), target).items():
                totals[prefix + "_" + name] = totals.get(prefix + "_" + name, 0.0) + value
        for name, value in mechanism_stats(out).items():
            stats_total[name] = stats_total.get(name, 0.0) + value
        n += 1
    return {k: v / n for k, v in totals.items()}, {k: v / n for k, v in stats_total.items()}


def main(stage="det"):
    print("===> Init settings")
    args = get_args(stage)
    set_random_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    out_dir = Path(args.output_dir)
    ckpt_dir = out_dir / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logger_test = get_logger("test", str(out_dir))
    tb_writer = SummaryWriter(str(out_dir / "tensorboard"))

    print("===> Loading datasets")
    train_set = Training_Dataset(args.dataset_dir, args.train_path, args.patch_size, args.repeat,
                                 args.exposure_file_name, args.label_file_name)
    test_set = Test_Dataset(args.dataset_dir, args.test_path, args.exposure_file_name, args.label_file_name)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                              pin_memory=True, drop_last=True, persistent_workers=args.num_workers > 0)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=args.test_num_workers)
    print("===> Training dataset size: {}, Testing dataset size: {}.".format(len(train_set), len(test_set)))

    print("===> Building model")
    set_compile(args.compile and use_cuda)
    model_kwargs = {}                              # allocation of the paper: D = eta Delta, eta = m (1 - e_bar)
    if args.stage == "comp":                       # same structure as the det checkpoint
        init = torch.load(args.init, map_location="cpu", weights_only=False)
        model_kwargs = model_options(init.get("config", {}).get("model", model_kwargs))
    model = OSHDRNet(**model_kwargs).to(device)
    if args.stage == "comp":
        model.load_state_dict(init["state_dict"], strict=True)
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith("odc."))
    trainable = [p for p in model.parameters() if p.requires_grad]

    # optimizer, lr_scheduler and loss
    optimizer = torch.optim.Adam(trainable, lr=args.lr, betas=(0.9, 0.999))
    steps_per_epoch = len(train_loader)
    scheduler = cosine_lr(optimizer, args.epochs * steps_per_epoch, args.eta_min)
    criterion = DetectionLoss(args.lambda_fft) if args.stage == "det" else CompletionLoss()

    # load checkpoint
    start_epoch = 1
    prefix = "final" if args.stage == "comp" else "det"
    if args.resume:
        print("===> Loading checkpoint from: {}".format(args.resume))
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(state["state_dict"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch = state["epoch"] + 1

    config = vars(args).copy()
    config.update({"model": model_kwargs, "parameters": sum(p.numel() for p in model.parameters()),
                   "parameters_by_part": {
                       "stems+RGA": sum(p.numel() for n, p in model.named_parameters()
                                        if n.startswith(("stem", "stages"))),
                       "DEA": sum(p.numel() for p in model.dea.parameters()),
                       "ODC": sum(p.numel() for p in model.odc.parameters())},
                   "trainable": sum(p.numel() for p in trainable), "steps_per_epoch": steps_per_epoch,
                   "train_samples": len(train_set), "test_scenes": len(test_set)})
    (out_dir / "config.json").write_text(json.dumps(config, indent=1))
    log = open(out_dir / "train.log", "a", buffering=1)
    log.write("CONFIG %s\n" % json.dumps(config))
    print("===> Parameters: {:,} (trainable {:,})".format(config["parameters"], config["trainable"]))
    use_odc = args.stage == "comp"

    for epoch in range(start_epoch, args.epochs + 1):
        print("===> Epoch: {}/{}, learning rate: {:1.7f}".format(epoch, args.epochs, scheduler.get_last_lr()[0]))
        tb_writer.add_scalar("train/lr", scheduler.get_last_lr()[0], epoch)
        train_stats = train_one_epoch(args, model, device, train_loader, optimizer, scheduler, criterion, epoch,
                                      log, tb_writer)
        summary = {"epoch": epoch, "lr": scheduler.get_last_lr()[0], "train": train_stats}
        # test-set evaluation for monitoring only (no checkpoint is selected with it)
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            if device.type == "cuda":
                torch.cuda.empty_cache()
            test, stats = evaluate(model, test_loader, device, args.test_patch_size, use_odc)
            if device.type == "cuda":
                torch.cuda.empty_cache()
            summary.update({"test": test, "mechanism": stats})
            for name, value in test.items():
                tb_writer.add_scalar("test/" + name, value, epoch)
            logger_test.info("Epoch {}: PSNR_mu: {:.4f}, PSNR_l: {:.4f}, SSIM_mu: {:.4f}, SSIM_l: {:.4f}".format(
                epoch, test[prefix + "_psnr_mu"], test[prefix + "_psnr_l"], test[prefix + "_ssim_mu"],
                test[prefix + "_ssim_l"]))

        # checkpoints at fixed epochs, and the training state to resume
        if epoch % args.save_every == 0 or epoch == args.epochs:
            torch.save({"epoch": epoch, "state_dict": model.state_dict(), "config": config},
                       ckpt_dir / ("epoch_%d.pth" % epoch))
        torch.save({"epoch": epoch, "state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "config": config},
                   ckpt_dir / "latest.pth")
        log.write("EPOCH %s\n" % json.dumps(summary))
        print(json.dumps(summary), flush=True)

    # the model of the last epoch is the result of this stage
    torch.save({"epoch": args.epochs, "state_dict": model.state_dict(), "config": config}, ckpt_dir / "final.pth")
    print("===> Final model saved to: {}".format(ckpt_dir / "final.pth"))
    log.close()
    tb_writer.close()


if __name__ == "__main__":
    main()

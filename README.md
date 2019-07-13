# A Modified reid_baseline supporting Multi-GPU and SyncBN training

Original repo [here](https://github.com/michuanhaohao/reid-strong-baseline)

However, the original repo uses [ignite](https://github.com/pytorch/ignite) for training and saving the model, which is incompatible with model using [SyncBN](https://github.com/vacancy/Synchronized-BatchNorm-PyTorch), So I reimplement the baseline without the use of ignite.

Most code in this repo is borrowed from the original one.

## Usage

1. Clone the repo using `git clone `
2. Compile the code for Cython accelerated evaluation code `cd evaluate/eval_cylib && make`
3. the [SyncBN](https://github.com/vacancy/Synchronized-BatchNorm-PyTorch) module is pure pytorch implementation, so no need to compile once you have pytorch.
4. Modify the training config in configs folder.
5. Start training:

```bash
# training with only one GPU
CUDA_VISIBLE_DEVICES=1 python main.py -c configs/debug.yml
# testing with one GPU
CUDA_VISIBLE_DEVICES=1 python main.py -t -c configs/debug.yml TEST.WEIGHT /path/to/saved/weights

# training with multi-GPU
CUDA_VISIBLE_DEVICES=1,2 python main.py -c configs/debug_multi-gpu.yml
# testing with multi-GPU
CUDA_VISIBLE_DEVICES=1,2 python main.py -t -c configs/debug_multi-gpu.yml TEST.WEIGHT /path/to/saved/weights

# fp16 training
CUDA_VISIBLE_DEVICES=1 python main.py -c configs/debug.yml SOLVER.FP16 True
```

## Result

I only trained the model with 30 epoch, so the model may not be fully converged.

**Note that the Resnet50 model is trained with Warmup, Random erasing augmentation, Last stride=1 and BNNeck.** 

### Result training with one GPU

```bash
2019-07-06 13:43:29,884 reid_baseline.eval INFO: Validation Result:
2019-07-06 13:43:29,885 reid_baseline.eval INFO: CMC Rank-1: 89.55%
2019-07-06 13:43:29,885 reid_baseline.eval INFO: CMC Rank-5: 96.76%
2019-07-06 13:43:29,885 reid_baseline.eval INFO: CMC Rank-10: 98.28%
2019-07-06 13:43:29,885 reid_baseline.eval INFO: mAP: 74.50%
2019-07-06 13:43:29,885 reid_baseline.eval INFO: --------------------
2019-07-06 13:45:40,783 reid_baseline.eval INFO: ReRanking Result:
2019-07-06 13:45:40,783 reid_baseline.eval INFO: CMC Rank-1: 92.67%
2019-07-06 13:45:40,783 reid_baseline.eval INFO: CMC Rank-5: 96.41%
2019-07-06 13:45:40,784 reid_baseline.eval INFO: CMC Rank-10: 97.48%
2019-07-06 13:45:40,784 reid_baseline.eval INFO: mAP: 90.00%
2019-07-06 13:45:40,784 reid_baseline.eval INFO: --------------------
```

### Result training with multi-GPU

```bash
2019-07-06 14:24:32,171 reid_baseline.eval INFO: Use multi gpu to inference
2019-07-06 14:25:41,053 reid_baseline.eval INFO: Validation Result:
2019-07-06 14:25:41,054 reid_baseline.eval INFO: CMC Rank-1: 87.44%
2019-07-06 14:25:41,054 reid_baseline.eval INFO: CMC Rank-5: 96.17%
2019-07-06 14:25:41,054 reid_baseline.eval INFO: CMC Rank-10: 97.54%
2019-07-06 14:25:41,054 reid_baseline.eval INFO: mAP: 72.13%
2019-07-06 14:25:41,054 reid_baseline.eval INFO: --------------------
2019-07-06 14:27:23,449 reid_baseline.eval INFO: ReRanking Result:
2019-07-06 14:27:23,450 reid_baseline.eval INFO: CMC Rank-1: 91.30%
2019-07-06 14:27:23,450 reid_baseline.eval INFO: CMC Rank-5: 95.64%
2019-07-06 14:27:23,450 reid_baseline.eval INFO: CMC Rank-10: 96.79%
2019-07-06 14:27:23,450 reid_baseline.eval INFO: mAP: 88.92%
2019-07-06 14:27:23,450 reid_baseline.eval INFO: --------------------
```

### Result training with FP16

```bash
2019-07-07 12:29:20,649 reid_baseline.eval INFO: Validation Result:
2019-07-07 12:29:20,650 reid_baseline.eval INFO: CMC Rank-1: 90.44%
2019-07-07 12:29:20,650 reid_baseline.eval INFO: CMC Rank-5: 96.47%
2019-07-07 12:29:20,650 reid_baseline.eval INFO: CMC Rank-10: 98.13%
2019-07-07 12:29:20,650 reid_baseline.eval INFO: mAP: 75.88%
2019-07-07 12:29:20,650 reid_baseline.eval INFO: --------------------
2019-07-07 12:31:01,269 reid_baseline.eval INFO: ReRanking Result:
2019-07-07 12:31:01,270 reid_baseline.eval INFO: CMC Rank-1: 93.14%
2019-07-07 12:31:01,270 reid_baseline.eval INFO: CMC Rank-5: 96.26%
2019-07-07 12:31:01,270 reid_baseline.eval INFO: CMC Rank-10: 97.12%
2019-07-07 12:31:01,270 reid_baseline.eval INFO: mAP: 90.22%
2019-07-07 12:31:01,270 reid_baseline.eval INFO: --------------------
```

### Result training on Market1501, testing on Partial-iLIDS

```bash
2019-07-08 18:15:14,299 reid_baseline.eval INFO: Validation Result:
2019-07-08 18:15:14,299 reid_baseline.eval INFO: CMC Rank-1: 45.38%
2019-07-08 18:15:14,300 reid_baseline.eval INFO: CMC Rank-3: 57.14%
2019-07-08 18:15:14,300 reid_baseline.eval INFO: mAP: 55.76%
2019-07-08 18:15:14,300 reid_baseline.eval INFO: --------------------
2019-07-08 18:15:14,678 reid_baseline.eval INFO: ReRanking Result:
2019-07-08 18:15:14,678 reid_baseline.eval INFO: CMC Rank-1: 28.57%
2019-07-08 18:15:14,678 reid_baseline.eval INFO: CMC Rank-3: 48.74%
2019-07-08 18:15:14,678 reid_baseline.eval INFO: mAP: 42.62%
2019-07-08 18:15:14,679 reid_baseline.eval INFO: --------------------
```

### Result on [VeRI-WILD](https://github.com/PKU-IMRE/VERI-Wild) small test set

```bash
2019-07-12 22:47:57,250 reid_baseline.eval INFO: Validation Result:
2019-07-12 22:47:57,250 reid_baseline.eval INFO: CMC Rank-1: 90.43%
2019-07-12 22:47:57,250 reid_baseline.eval INFO: CMC Rank-5: 96.85%
2019-07-12 22:47:57,250 reid_baseline.eval INFO: CMC Rank-10: 98.19%
2019-07-12 22:47:57,250 reid_baseline.eval INFO: mAP: 74.27%
2019-07-12 22:47:57,251 reid_baseline.eval INFO: --------------------
2019-07-12 22:57:10,006 reid_baseline.eval INFO: ReRanking Result:
2019-07-12 22:57:10,006 reid_baseline.eval INFO: CMC Rank-1: 89.66%
2019-07-12 22:57:10,006 reid_baseline.eval INFO: CMC Rank-5: 95.45%
2019-07-12 22:57:10,006 reid_baseline.eval INFO: CMC Rank-10: 97.46%
2019-07-12 22:57:10,007 reid_baseline.eval INFO: mAP: 77.42%
2019-07-12 22:57:10,007 reid_baseline.eval INFO: --------------------
```

### Result on [VeRI776](https://github.com/VehicleReId/VeRidataset)

```bash
2019-07-13 11:35:12,402 reid_baseline.eval INFO: Validation Result:
2019-07-13 11:35:12,402 reid_baseline.eval INFO: CMC Rank-1: 95.65%
2019-07-13 11:35:12,402 reid_baseline.eval INFO: CMC Rank-5: 97.91%
2019-07-13 11:35:12,403 reid_baseline.eval INFO: CMC Rank-10: 99.11%
2019-07-13 11:35:12,403 reid_baseline.eval INFO: mAP: 77.15%
2019-07-13 11:35:12,403 reid_baseline.eval INFO: --------------------
2019-07-13 11:36:02,050 reid_baseline.eval INFO: ReRanking Result:
2019-07-13 11:36:02,050 reid_baseline.eval INFO: CMC Rank-1: 97.20%
2019-07-13 11:36:02,050 reid_baseline.eval INFO: CMC Rank-5: 97.85%
2019-07-13 11:36:02,050 reid_baseline.eval INFO: CMC Rank-10: 98.75%
2019-07-13 11:36:02,050 reid_baseline.eval INFO: mAP: 81.74%
2019-07-13 11:36:02,050 reid_baseline.eval INFO: --------------------
```

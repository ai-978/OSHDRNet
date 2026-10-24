# -*- coding:utf-8 -*-
"""Training of OSHDRNet, stage 2 (paper Sec. III-F): the RGA and DEA of stage 1 (python train.py) are frozen
and the completion network C_theta of ODC is trained with L_comp on the completion regions they produce.
The options are those of train.py (stage-2 log directory and evaluation interval, see train.STAGES).
"""
from train import main


if __name__ == "__main__":
    main(stage="comp")

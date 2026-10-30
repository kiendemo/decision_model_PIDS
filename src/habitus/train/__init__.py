"""Trainers of the decision models (GPU; not needed to reproduce the main numbers).

    calibrate   the shared loss (soft_ce), inner-day temperatures and calibration reports
    jev         LoRA / LoRA + letter head on a decoder backbone (train_jev)
    laya        Laya encoder seeds and ensembles (train_laya), laya_mq the multi-question layout
    items       label-free habit items exported from the training hosts' clean days

Importing this package imports nothing else (no torch).
"""

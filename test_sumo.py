import os
import subprocess


for ckpt in range(1, 400):
    command = f"python display.py --cfg config/SG/robo-sumo-sgant-ant-v0.yaml --ckpt {ckpt} --ckpt_dir tmp/models"
    os.system(command)

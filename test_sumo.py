import os
import subprocess


for ckpt in range(1, 189):
    command = f"python display.py --cfg config/robo-sumo-sgdevant-devant-v0.yaml --ckpt {ckpt} --ckpt_dir tmp/models"
    os.system(command)

import os
import subprocess


for ckpt in range(1, 200):
    command = f"python display.py --cfg config/test-robo-sumo-sgdevspider-devspider-v0.yaml --ckpt {ckpt} --ckpt_dir tmp/models"
    os.system(command)

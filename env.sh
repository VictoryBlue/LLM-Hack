#!/bin/bash

pip install -U huggingface_hub

pip install -r requirements.txt

HF_ENDPOINT=https://hf-mirror.com python main.py 2>&1 | tee output.log
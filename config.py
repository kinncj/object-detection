# Copyright (c) 2024 Kinn Coelho Juliao <kinncj@gmail.com>
# All rights reserved.
#
# This software is licensed under the terms of the MIT License.
# See the LICENSE file in the project root for license terms.
import os
import torch

# Configuration
TEMP_DIR = "/tmp/ai_files"
os.makedirs(TEMP_DIR, exist_ok=True)

# Device configuration
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Restricted classes and their colors
RESTRICTED_CLASSES = {
    1: "person",
    77: "cell phone",
    73: "laptop",
    72: "tv",
    76: "keyboard",
    74: "mouse"
}

RESTRICTED_COLORS = {
    "person": (0, 0, 255),
    "cell phone": (0, 255, 0),
    "laptop": (255, 255, 0),
    "tv": (0, 165, 255),
    "keyboard": (255, 0, 255),
    "mouse": (255, 128, 0)
}
#!/bin/bash
ffmpeg -framerate 30 -pattern_type glob -i "frames/combined_*.png" -c:v libx264 -pix_fmt yuv420p video.mp4
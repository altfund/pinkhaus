#!/bin/bash
cd "/home/ess/Documents/apps/ominari/projects/ominari"
"/home/ess/.local/Spyder-5.4.5/envs/spyder-5.4.5/bin/python" "run_everything.py" >> scheduler_output.log 2>> scheduler_error.log

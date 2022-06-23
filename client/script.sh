#!/bin/bash
var=`id -Gn $USER`
python3 video_screen.py --name $USER --group $var

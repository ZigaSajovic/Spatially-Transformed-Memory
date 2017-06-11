#!/bin/bash
cd tmp_
convert -delay 20 -loop 0 *.png "$1".gif
rm *.png
cd ..
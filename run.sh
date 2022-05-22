#!/bin/bash
# train the model
echo "Training the model..."
python train.py
echo "Training completed!"
# test the model 
echo "Testing the model"
python test.py  --color_space True --color_component True  --texture True
echo "Testing completed!"
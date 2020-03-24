This method is a modification of the MARS(https://github.com/craston/MARS) method.
Most of the operations are the same as MARS
This experiment mainly achieves RGB and Flow  deep mutual learning based on the KL divergence.


## Training script
For  and UCF101
 ```
    python MARS_train.py --dataset UCF101 --modality RGB_Flow --split 1  --n_classes 101 \
--n_finetune_classes 101 --batch_size 64 --log 1 --sample_duration 16 --model resnext \
--model_depth 101 --ft_begin_index 4 --output_layers 'avgpool' --MARS_alpha 50 --frame_dir  "dataset/UCF101" \
--annotation_path "/dataset/UCF101_labels/" --pretrain_path "/dataset/UCF101/RGB_UCF101_16f.pth"  \
--Flow_resume_path "/dataset/UCF101/Flow_UCF101_16f.pth"
```
We use pre-trained weights for RGB and FLOW for experiments
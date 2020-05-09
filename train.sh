checkpoint="/cmlscratch/arjgpt27/projects/ENPM673/DL"
modelpath="${checkpoint}/alexnet_seed_${seed}_normalize=True_augment=False_optimizer=SGD_epoch=1.t7"
output_dir="alexnet_chkpoints"

echo " "
echo "mkdir -p ${output_dir}"
mkdir -p $output_dir
echo " "
echo "Training AlexNet."
python train_model.py --seed 40 --model alexnet --epochs 40 --lr 0.1 --lr_factor 0.5 --lr_schedule 5 15 25 --optimizer SGD --checkpoint $checkpoint --save_net --output $output_dir --normalize
# python test_model.py --model alexnet --normalize --output $output_dir --model_path $modelpath
echo "Training and testing complete."
echo " "
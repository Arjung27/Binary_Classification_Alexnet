# Binary_Classification_Alexnet
Binary classification between dogs and cats using alexnet inspired architecture

## Dependencies
```
python3, numpy, matplotlib
OpenCV
CUDA 10.0 #tested with this particular version
Pytorch 1.4 
```
## Run the code
* Clone the repository
* To understand the data
```
python3 data_statistics.py --data=<path-to-data>
```
* Data preperation
```
python3 data_prep.py --data=<path-to-data> --output=<path-to-output>
```
* Running the code
```
sh train.sh
```

## Results
* Training Loss
![Training Loss](https://github.com/Arjung27/Binary_Classification_Alexnet/blob/master/plots/train_loss.png)
* Testing Loss
![Testing Loss](https://github.com/Arjung27/Binary_Classification_Alexnet/blob/master/plots/test_loss.png)
* Training Accuracy
![Training Accuracy](https://github.com/Arjung27/Binary_Classification_Alexnet/blob/master/plots/training_acc.png)
* Testing Accuracy
![Testing Accuracy](https://github.com/Arjung27/Binary_Classification_Alexnet/blob/master/plots/test_acc.png)

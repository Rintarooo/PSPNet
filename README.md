# Segmentation for CityScapes dataset

Straightforward implementation of Segmentation for CityScapes dataset

## Description
  
First, you can go to the official site(https://www.cityscapes-dataset.com/downloads/).
  
Then, download `leftImg8bit_trainvaltest.zip`(11GB) and `gtFine_trainvaltest.zip`(241MB). Make sure that you've registered your email on this site before downloading and that email is not a gmail account. 

I personally used 2975 images as train data, 500 images as both validation and test data. I saved configuration details in files located at `Csv` dir. 

Pyramid Scene Parsing Network (PSPNet) is the model architecture proposed by this paper(https://arxiv.org/abs/1612.01105).

  
<img src="https://user-images.githubusercontent.com/51239551/98388785-8ac0b380-2096-11eb-8a61-44401b1ec8b6.png" width="230"/> <img src="https://user-images.githubusercontent.com/51239551/98388903-af1c9000-2096-11eb-88bf-fd5ce39b1d2c.png" width="230"/> <img src="https://user-images.githubusercontent.com/51239551/98388922-b5ab0780-2096-11eb-920b-f768001eb05e.png" width="230"/>

  

Input Image(left), Output Image(center), Ground Truth(right)

## Usage
  
generate the pickle file contaning hyperparameter values by running the following command.

```
python config.py (-h)
```

you will see the pickle file in `Pkl` dir.  
now you can start training the model.

```
python train.py -p Pkl/***.pkl
```

After training is done, you could see prediction of pretrained model

```
python infer.py -p Weights/***.pt
```

### Dependencies
* Python = 3.6.10
* PyTorch = 1.6.0
* OpenCV (pip install opencv-python)
* numpy
* matplotlib (only for test)

## Environment
I leave my own environment below. I tested it out on a single GPU.

 PSPNet is a kind of large model architecture, so the bigger GPU memory is desirable for training.  
  

* OS:
	* Linux(Ubuntu 18.04.5 LTS) 
* GPU:
	* NVIDIA® GeForce® RTX 2080 Ti VENTUS 11GB OC
* CPU:
	* Intel® Xeon® CPU E5640 @ 2.67GHz

## Reference
* https://github.com/fregu856/deeplabv3
* https://github.com/fregu856/segmentation
* https://github.com/YutaroOgawa/pytorch_advanced/tree/master/3_semantic_segmentation (written in Japanese)

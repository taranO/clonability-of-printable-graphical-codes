## Clonability of anti-counterfeiting printable graphical codes: a machine learning approach

The research was supported by the [SNF](http://www.snf.ch) project No. 200021_182063. 
##

PyTorch implementation of ["Clonability of anti-counterfeiting printable graphical codes:
a machine learning approach"](http://sip.unige.ch/projects/snf-it-dis/publications/icassp-2019) 

The security of printable codes in terms of their reproducibility by unauthorized parties or clonability is largely unexplored. We try to investigate the clonability of printable graphical codes from a machine learning perspective. The proposed framework is based on a simple system composed of fully connected neural network layers. The results obtained on real codes printed by several printers demonstrate a possibility to accurately estimate digital codes from their printed counterparts in certain cases. This provides a new insight on scenarios, where printable graphical codes can be accurately cloned.

<p align="center">
<img src="http://sip.unige.ch/files/2815/5291/8110/2019_icassp_training_procedure.png" width="450px" align="center">
<br/>
<br/>
Fig.1: Generalized diagram of training procedure.  
</p>

The main goal of this work is to investigate the resistance of PGC to clonability attacks. The overwhelming majority of such attacks can be split into two main groups: (a) handcrafted attacks, which are based on the experience and knowhow of the attackers and (b) machine learning based attacks, which use training data to create clones of the original codes.

In our work, we focus on the investigation of machine learning based attacks due to the recent advent in the theory and practice of machine learning tools. Growing popularity and remarkable results of deep neural network (DNN) architectures in computer vision applications motivated us to investigate the clonability of PGC using these architectures trained for different classes of printers.

The main contributions are: 
* we investigate the clonability of printable graphical codes using machine learning based attacks;
* we examine the proposed framework on real printed codes reproduced with 4 printers;
* we empirically demonstrate a possibility to sufficiently accurately clone the PGC from their printed counterparts in certain cases.

To provide more understanding how the codes look, we visualize the sub-blocks of size 84 × 84 from several codes for each printer and the estimations deploying the BN as the best estimator in Fig. 2.
 
<p align="center">
<img src="http://sip.unige.ch/files/7215/5291/8832/2019_icassp_002.png" width="750px" align="center">
<br/>
<br/>
Fig.2: Examples of attacks against PGC: two samples of scanned codes, the estimates produced by BN model
and the difference between the original and estimated codes..  
</p>

## Requirements 
* numpy
* scipy
* matplotlib
* pytorch-gpu
* torchvision-gpu

## Dataset

The dataset is available at https://cuicloud.unige.ch/index.php/s/t65MFSNrS4dmMTQ

## Train 

    $ python train_model.py --model_type=bn --code=sa --n_epochs=1000
    
The threshold estimation on the validation sub-set:     

       $ python test_thr.py --model_type=bn --code=sa --n_epochs=1000
       
## Test

    $ python codes_regeneration.py --model_type=bn --code=sa --epoch=1000 --thr=0.5


## Citation
O. Taran, S. Bonev, and S. Voloshynovskiy, "Clonability of anti-counterfeiting printable graphical codes: a machine learning approach," in Proc. IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), Brighton, United Kingdom, 2019. 
  
    @inproceedings { Taran2019icassp,
      author = { Taran, Olga and Bonev, Slavi and Voloshynovskiy, Slava },
      booktitle = { IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) },
      title = { Clonability of anti-counterfeiting printable graphical codes: a machine learning approach },
      address = { Brighton, United Kingdom },
      month = { May },
      year = { 2019 }
    }

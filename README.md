Learning Dual Modality Interactions for Event-based Motion Deblurring
====
Zeyu Xiao, Zhuoyuan Li, Yang Zhao, Yu Liu, Zhao Zhang, and Wei Jia. [Learning Dual Modality Interactions for Event-based Motion Deblurring](xxx). In submission. <br/>


### Installation
This implementation based on [BasicSR](https://github.com/xinntao/BasicSR) which is a open source toolbox for image/video restoration tasks. 

```python
python 3.8.5
pytorch 1.7.1
cuda 11.0
```


### Train
---
#### GoPro

* prepare data
  
  * download the GoPro events dataset (see [Dataset](dataset_section)) to 
    ```bash
    ./datasets
    ```

  * it should be like:
  
    ```bash
    ./datasets/
    ./datasets/DATASET_NAME/
    ./datasets/DATASET_NAME/train/
    ./datasets/DATASET_NAME/test/
    ```

* train

  * ```python basicsr/train.py -opt options/train/GoPro/DuIntNet.yml ```

* eval
  * ```python basicsr/test.py -opt options/test/GoPro/test_DuIntNet.yml  ```
  

#### REBlur

* prepare data
  
  * download the REBlur dataset (see [Dataset](dataset_section)) to 
    ```bash
    ./datasets
    ```

  * it should be like:
  
    ```bash
    ./datasets/
    ./datasets/DATASET_NAME/
    ./datasets/DATASET_NAME/train/
    ./datasets/DATASET_NAME/test/
    ```

* finetune
  * ```python ./basicsr/train.py -opt options/train/REBlur/DuIntNet_reblur.yml```

* eval
  * ```python basicsr/test.py -opt options/test/REBlur/test_DuIntNet.yml ```
  


## Citation
```
@article{xiao2025learning,
  title={Learning Dual Modality Interactions for Event-based Motion Deblurring},
  author={Xiao, Zeyu and Li, Zhuoyuan and Zhao, Yang and Liu, Yu and Zhang, Zhao and Jia, Wei},
  journal={In submission},
  year={2025}
}
```

## Contact
Any question regarding this work can be addressed to zeyuxiao1997@163.com and 17756311760 (Wechat|微信).

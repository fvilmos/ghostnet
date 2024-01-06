### GhostNet implementation in Keras

This work implements the building blocks and network of the GhostNet [1] in tensorflow keras. It follows the logic presented in [4] with some adaptions towards the used machine learning framework.  

### Network configuration

The authors of the GhostNet were inspired by MobilenetV3 [2] architecture, this work is implemented 2 architectures:
- "large" that is according to the GhostNet paper (and follows the structure of ```MobilenetV3-Large```).  

- "small" folows the ```MobileNetV3-Small``` from MobilenetV3 paper.

Architecture selection (use "small" or "large" tags in constructor):
```
# create the ghostnet model
ghost_net_model = ghost_net_obj.create_model("small")
```

The "width_multiplier" parameter can be used to scale up or down the GhostNet, which results in architectures with fewer parameters ```(i.e. width_multiplier=0.1) ~1.4M.```<br>

Block configuration and meaning: ```[16,16,0.25,2,3]``` - expansion_ratio, out_channels, Squeeze-Excitation[3] value, strides, kernel size.

```Note: the original paper used the standard 224x224 input, smaller inputs can be applied, if the AveragePooling kernel is adjusted, or adjusting the Squeeze-Excitation module value.```  

<p align="center"> 
  <img src="info/ghostnet_90ccw.png" alt="" width="2048"></a>
  <div align="center">GhostNet - small, SE=0.25</div>
</p>

#### Comparation, tests
|Model| Parameters [M]|FLOPS[G]|
|---|---|---|
|GhostNet (small) / SE - 0.25|1.47|0.103|
|ResNet [5] - custom|0.287|0.594|


### Resources
1. [GhostNet: More Features from Cheap Operations, Kai Han, et all](https://arxiv.org/abs/1911.11907)
2. [Searching for MobileNetV3, Andrew Howard, et all](https://arxiv.org/abs/1905.02244)
3. [Squeeze-and-Excitation Networks, Jie Hu, et all](https://arxiv.org/abs/1709.01507)
4. [HUAWEI Noah's Ark Lab, Efficient-AI-Backbones](https://github.com/huawei-noah/Efficient-AI-Backbones)
5. [fvilmos, ResNet_keras](https://github.com/fvilmos/ResNet_keras)

/Enjoy.
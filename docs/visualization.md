
## Overview of STANet architecture. 
![image](../results/STANet.png)
Overview of STANet architecture. The input is a time-series RGB image, and the output is the semantic segmentation mask of soybean canopy. (a) STANet is an encoder-decoder model. The encoder adopts ResNet10l as backbone network and innovatively introduces deformable convolution (DCN) and coordinate attention (CA) modules to extract multi-scale features. The decoder designs spatio-temporal alignment module (STAM) to extract multi-scale features of canopy. (b) The structure of CA improves feature discrimination ability, particularly at canopy boundaries. (c) The structure of STAM captures time-series dependencies.

## Overview of TLA architecture. 
![image](../results/TLA.png)
Overview of the TLA method. (a) The rotation angle (θ) of original canopy semantic segmentation map, which is measured using a protractor. (b) The results of rotated canopy semantic segmentation map. Generation of (c) horizontal traction lines (HTLs) and (d) vertical traction lines (VTLs). Optimization of (e) HTLs and (f) VTLs. The black circles represented intersected traction lines (TLs) need to be optimized. (g) The results of plot instance segmentation. The gray lines represent the preliminary TLs, the green lines represent the retained TLs of plot boundaries. The yellow lines represent the HTLs (before and after optimization), and the red lines represent the plot VTLs (before and after optimization). 

## The visualization results for canopy semantic segmentation at 37 days after sowing of STANet
![image](../results/Canopy-semantic-segmentation.png)

## The visualization results for plot instance segmentation at 37 days after sowing of TLA
![image](../results/Plot-instance-segmentation.png)

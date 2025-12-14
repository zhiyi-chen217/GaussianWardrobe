
# <b> Gaussian Wardrobe: </b>:Compositional 3D Gaussian Avatars for Free-Form Virtual Try-on

***Abstract**: We introduce Gaussian Wardrobe, a novel framework to digitalize compositional 3D neural avatars from multi-view videos.
Existing methods for 3D neural avatars typically treat the human body and clothing as an inseparable entity. 
However, this paradigm fails to capture the dynamics of complex free-form garments and limits the reuse of clothing across different individuals.
To overcome these problems, we develop a novel, compositional 3D Gaussian representation to build avatars from multiple layers of free-form garments.
The core of our method is decomposing neural avatars into bodies and layers of shape-agnostic neural garments.
To achieve this, our framework learns to disentangle each garment layer from multi-view videos and canonicalizes it into a shape-independent space.
In experiments, our method models photorealistic avatars with high-fidelity dynamics, achieving new state-of-the-art performance on novel pose synthesis benchmarks.
In addition, we demonstrate that the learned compositional garments contribute to a versatile digital wardrobe, enabling a practical virtual try-on application where clothing can be freely transferred to new subjects.


# Installation
0. Clone this repo.
1. Install environments.
```
# install requirements
pip install -r requirements.txt

# install diff-gaussian-rasterization-depth-alpha
cd gaussians/diff_gaussian_rasterization_depth_alpha
python setup.py install
cd ../..

# install styleunet
cd network/styleunet
python setup.py install
cd ../..
```
2. Download [SMPL-X](https://smpl-x.is.tue.mpg.de/download.php) model, and place pkl files to ```./smpl_files/smplx```.
# Data Preparation

## Customized Dataset

# Avatar Training


# Avatar Animation

# Virtual Try-on

3. Run:
```
python main_avatar.py -c configs/avatarrex_zzr/avatar.yaml --mode=test
```
You will see the animation results like below in `./test_results/avatarrex_zzr/avatar`.

https://github.com/lizhe00/AnimatableGaussians/assets/61936670/5aad39d2-2adb-4b7b-ab90-dea46240344a

# Evaluation


# Acknowledgement

# Citation
```


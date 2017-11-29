# statistical-learning-image-classification

## Objective
The goal is to segment the “cheetah” image (shown below in the left) into its two components, cheetah (foreground aka FG) and grass
(background aka BG)

| ![](data/test/cheetah.bmp) | ![](data/test/cheetah_mask.bmp) |
|:---:|:---:|
| Test Image | Test Image Ground Truth |

## Data pre-processing

To formulate this as a pattern recognition problem, we need to decide on an observation space. Here
we will be using the space of 8×8 image blocks, i.e. we view each image as a collection of 8×8 blocks.
For each block we compute the discrete cosine transform (function dct2 on MATLAB) and obtain
an array of 8 × 8 frequency coefficients. We do this because the cheetah and the grass have different
textures, with different frequency decompositions and the two classes should be better separated in the
frequency domain.

The file data/train/TrainingSamplesDCT_8.mat contains a training set of vectors obtained from a similar image (stored as a matrix, each row is a training vector) for each of the classes. There are two matrices, TrainsampleDCT BG and TrainsampleDCT FG for foreground
and background samples respectively

## Baye's Decision Rule

The Bayesian Decision Rule states the we have to pick the class i which minimizes the Risk associated
by picking the class.For the “0-1” loss the optimal decision rule is the maximum a-posteriori probability
rule.


Therefore, for a given input feature x, the value the class i ( cheetah or grass) which maximizes the
above equation is chosen as the output class. Accordingly the pixel value is coded as ’1’ (FG) or ’0’(BG).


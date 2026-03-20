# GERD: Geometric event response data generation

[![](https://img.shields.io/badge/DOI-arXiv%3A1408.3644-B31B1B.svg)](https://arxiv.org/abs/2412.03259)
[![](https://zenodo.org/badge/DOI/10.5281/zenodo.11063678.svg)](https://zenodo.org/records/11063678)

GERD generates event-based datasets for objects tracking tasks at arbitrary resolutions subject to arbitrary transformations.
Both the shapes and the transformations can be arbitrarily parameterized so you can carefully control the type of change happening to each object in the scene over time.
This is useful to study event-based algorithms from first-principles: objects in the real world are controlled by highly specific transformations, but that is often overlooked in the literature.

![](affine.gif)

Example render of shapes subject to affine transformation with a relatively high velocity (v=2.56).

## Important simulation details (read me first)
We apply translation, scaling, rotation, and shearing to the shapes independently.
You can configure the transformations to use different starting conditions and the velocities will be updated according to a specified [PyTorch distribution](https://pytorch.org/docs/stable/distributions.html).
All of this is parameterized in the `RenderParameters` class in the [`render.py`](gerd/render.py) file.

### Activation normalization
Translation velocities are normalized to the pixel grid, meaning that a velocity of 1 in the x axis means that the object moves one pixel to the right every frame.
The other velocities are normalized to produce a similar number of pixel activations, to avoid skewing the dataset towards a specific transformation.

### Fractional velocities and upsampling
A velocity of 0.1 is problematic in a pixel grid, so upsample to a grid that, by default, is 8 times larger than the specified resolution.
An event in the downsampled (actual) grid will "trigger" when a certain fraction of the upsampled pixels are turned on.
To accumulate pixel activations in the upsampled grid over time, we use a thresholded integrator.

## Usage

You can install GERD by running 
`pip install gerd` or by manually pulling the repository and installing the local version with `pip install <path-to-gerd>`.

The code is written in Python using the [PyTorch](https://pytorch.org/) library.
GERD roughly has two use cases: 1) generate datasets for training or simulation and 2) render a few and carefully crafted simulations for visualizations or surgical tests

### 1. Generating datasets
To generate a dataset, use the `gerd` command (see `gerd --help` for more information).
The example below generates 1000 videos that translates and scales into the `/data` directory.

1. Configure the dataset parameters
    * Copy and modify the `example.yaml` configuration to suit your needs
2. Run the dataset generation
    * `gerd 1000 /data example.yaml`
    * Note that the data is saved as a [sparse PyTorch tensor](https://pytorch.org/docs/stable/sparse.html)
3. Import and use the dataset in your training code as a [PyTorch dataset](https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html)
```python
import gerd

my_train_dataset = gerd.GerdDataset("/data", train=True)
my_test_dataset = gerd.GerdDataset("/data", train=False)
```

By default, the dataset will crop the frames to 40 timesteps and assume that each file contains 128 timesteps.
You can change this by providing additional parameters to the `GerdDataset` class.

### 2. Render precise simulations
On a low level, we offer a general generating function `render` in the [`render.py`](gerd/render.py) file, that can render specific shapes, defined in [`shapes.py`](gerd/shapes.py).

On a higher level, the [`main.py`](gerd/main.py) file contains a script that generates a dataset of three specific objects moving in a scene: a square, a circle, and a triangle.
We will cover that usecase below:


## Authors

* [Jens E. Pedersen](https://www.kth.se/profile/jeped) (@GitHub [jegp](https://github.com/jegp/)), doctoral student at KTH Royal Institute of Technology, Sweden.
* [Dimitris Korakounis](https://www.kth.se/profile/dimkor), doctoral student at KTH Royal Institute of Technology, Sweden.
* [Raghav Singhal](https://github.com/RaghavSinghal10), visiting student at KTH Royal Institute of Technology, Sweden.
* [Jörg Conradt](http://neurocomputing.systems/), principal investigator

The work has received funding from the EC Horizon 2020 Framework Programme under Grant Agreements 785907 and 945539.

## Citation

If you use this work, please cite it as follows

```
@misc{pedersen2024gerd,
      title={GERD: Geometric event response data generation}, 
      author={Jens Egholm Pedersen and Dimitris Korakovounis and Jörg Conradt},
      year={2024},
      eprint={2412.03259},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2412.03259}, 
}
```

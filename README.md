# Moving MNIST Clutter

Generates a collection of movie featurs MNIST numbers moving around the screen. One is moving in a straight line, while the move randomly.

### Setting up environment

> conda env create -f environment.yml`

You may encounter an error with the `pillow` package that is accessed by `torchvision`. If so, take the following steps:
1. Activate the environment.
2. Uninstall `pillow`
    > pip uninstall pillow
3. Reinstall `pillow`
    > pip install pillow

You may encounter an error with the `matplotlib` package. If so, take the following steps:
1. Activate the environment.
2. Uninstall `matplotlib`
    > pip uninstall matplotlib
3. Reinstall `matplotlib`
    > pip install matplotlib
# BundleAdjustment

Modelling and solving bundle adjustment problems in Julia.

Given a set of images depicting a number of 3D points from different viewpoints, bundle adjustment can be defined as the problem of simultaneously refining the 3D coordinates describing the scene geometry and the parameters of the cameras (relative motion and optical characteristics) employed to acquire the images (https://en.wikipedia.org/wiki/Bundle_adjustment).

Mathematically, it amounts to solving the following non-linear least square problem:

```julia
  min    ∑ᵢ ∑ⱼ  vᵢⱼ (P(Xᵢ, Cⱼ) - xᵢⱼ)²
Xᵢ, Cⱼ
```

where:
- each Xᵢ is the vector containing the 3D coordinates of point number i,
- each Cⱼ is the vector containing the parameters of camera number j, 
- vᵢⱼ equals to 1 if point i has been observed by camera j, and equals 0 otherwise, 
- P(Xᵢ, Cⱼ) is the projection of point Xᵢ on camera Cⱼ (as defined in https://grail.cs.washington.edu/projects/bal/),
- xᵢⱼ is the vector of the 2D coordinates of the observation of point i on camera j. 

The data I use can be found here: https://grail.cs.washington.edu/projects/bal/. They are seperated into 5 datasets: 
- Ladybug (https://grail.cs.washington.edu/projects/bal/ladybug.html),
- Trafalgar Square (https://grail.cs.washington.edu/projects/bal/trafalgar.html), 
- Dubrovnik (https://grail.cs.washington.edu/projects/bal/dubrovnik.html), 
- Venice (https://grail.cs.washington.edu/projects/bal/venice.html),
- Final (https://grail.cs.washington.edu/projects/bal/final.html). 

Thus to run my code, you need to create a Data/ folder at the project root and to create five subfolders (LadyBug, Trafalgar, Dubrovnik, Venice and Final) and to download the .bz2 files in the corresponding subfolders.

Or, alternativly, you can run the get_data.sh script that will create the Data/ folder and all his sub-folders, and that will download all the datasets for you.


Python script (to have a some results to compare mine with) that export the vector of resiuals as .txt and solves bundle adjustment problems using the least_square function from scipy.optimize (code from https://scipy-cookbook.readthedocs.io/items/bundle_adjustment.html):
https://colab.research.google.com/drive/1li9dBcQ-9feva89QmYI16qgiEBFCf8L9

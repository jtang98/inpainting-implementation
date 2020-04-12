# inpainting-implementation

- The *patch_inpainting.py* file is an implementation of the paper **Object Removal by Exemplar-Based Inpainting** (A. Criminisi, P. Pérez, K. Toyama).
- The *energy_minimization_inpainting.py* file is an implementation of the paper **Non-Local Patch-Based Image Inpainting** (Alasdair Newson, Andrés Almansa, Yann Gousseau, Patrick Pérez).

The second algorithm uses the output of the first one as its initialization, which is very important, since the problem is highly non-convex.

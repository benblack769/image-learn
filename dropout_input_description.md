### dropout input

The idea is this:

1. Use a global comparator task as a dummy task.
2. Use dynamic dropout to estimate how much each input activation contributes to the overall dummy task
3. Dynamically update input activation to drop out best performing inputs with greater probability than low performing inputs. Very well performing inputs should drop with 100% prob, and very poor inputs should drop with 0% prob.
4. Make sure to add in which inputs were dropped into the input
5. The output of the network is the input drop weight (prob of being dropped)
6. Construct a reasonably deep network (maybe 2 or 3 hidden layers) with binary multiplicative activation functions. These should capture the highly interactive nature of the global comparator problem much better than unary operations.

To evaluate this network, use the drop weight to try to reconstruct the image.

https://infoscience.epfl.ch/record/174677/files/EPFL_TH5310.pdf

Interesting paper dealing with capsule dropout: "Sparse Unsupervised Capsules Generalize Better" https://arxiv.org/pdf/1804.06094.pdf

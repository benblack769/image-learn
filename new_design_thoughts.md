So, thoughts:

Main idea is to try to create a missing information problem by removing almost all the information, and allowing the network to decide which new information to reveal. Information is revealed through changes in a mask. This mask can operate on input pixels, input channels, network capsules (part of a layer), etc.

The network will decide what information to reveal next using a traditional deep net with skip connections, and a soft max followed by sampling. This recurrent process will be trained with 

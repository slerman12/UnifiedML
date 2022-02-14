from torch import nn

import Utils

from Blocks.Architectures.LermanBlocks.BioNet.NonLocalityCNN import NonLocalityCNN
from Blocks.Architectures.LermanBlocks.BioNet.LocalityViT import LocalityViT
from Blocks.Architectures.MultiHeadAttention import CrossAttentionBlock, SelfAttentionBlock


class BioNet(nn.Module):
    """Disentangling "What" And "Where" Pathways In CNNs"""

    def __init__(self, input_shape, out_channels, depth=3, output_dim=128):
        super().__init__()
        in_channels = input_shape[0]
        self.output_dim = output_dim

        self.ventral_stream = NonLocalityCNN(in_channels, out_channels, depth=depth)
        self.dorsal_stream = LocalityViT(input_shape, out_channels, depth)

        self.cross_talk = [CrossAttentionBlock(dim=out_channels, heads=8)
                           for _ in range(depth)]

        self.repr = nn.Sequential(Utils.ChannelSwap(),
                                  SelfAttentionBlock(dim=out_channels, heads=8),
                                  Utils.ChannelSwap(),  # Todo just use einops rearange
                                  nn.AdaptiveAvgPool2d(output_dim ** 0.5),
                                  nn.Flatten())

    def output_shape(self, h, w):
        return 1, self.output_dim

    def forward(self, input):
        ventral = self.ventral_stream.trunk(input)
        dorsal = self.dorsal_stream.trunk(input)

        t = Utils.ChannelSwap()

        for what, where, talk in zip(self.ventral_stream.CNN,
                                     self.dorsal_stream.ViT,
                                     self.cross_talk):
            ventral = what(ventral)
            dorsal = t(talk(t(where(dorsal)),
                            t(ventral).view(*t(ventral).shape[:-1], 2, -1)))  # Feature redundancy(? till convolved)

            # if self_supervise:
            #     loss = t(byol(talk2(t(ventral).view(*t(ventral).shape[:-1], 2, -1)), t(dorsal)), t(dorsal).mean(-1))
            #     Utils.optimize(loss,
            #                    self)

        out = self.repr(dorsal)
        return out

    """Aside from the way the modules are put together (via two disentangled streams), 
    there is not much that we have not seen before in some form
    
    We have a CNN [resnet, convmixer], ViT [worth 1000 words], CrossAttention [perceiver], 
    SelfAttention [all you need], and average pooling [pool].
    
    Here is a schematic visualization for those unfamiliar with Pytorch:
    
    [eye,, self attend + average <- CNNs,: ViTs <- input
    
    Now, let's go over the specifics of the CNN and ViT. 
    
    Let's overview a Patched Ftheta for any neural network function Ftheta.
    
    Given a grid of arbitrary dimensions, 
    we initialize a uniquely parameterized Ftheta for each designated-size patch
    
    Ftheta1(Patch 1), Ftheta2(Patch 2), ...,
    
    within the grid.
    
    These grid elements are then projected to the output dimensionality of Ftheta, preserving grid ordering.
    
    Unlike convolutions, this operation is intrinsically localized due to parameter non-sharing, 
    and therefore locality embeddings are not needed. 
    However the operation is efficient compared to a fully-connected MLP, much like a CNN.
    
    We can then repeat this for each subsequent grid layer up to chosen depth.
    
    This is the basic variant. We divide the dorsal input-grid into N equal-sized patches along height and width. 
    Each layer is topographically isotropic to the previous.
    
    On the ventral side, a configurable depth-size of residual blocks non-locally transforms the image input.
    Between each block, information is passed to the same-depth layer of the dorsal stream via cross attention. 
    Because this information is one-way, non-locality remains uncompromised. 
    Because of the nature of cross attention, a biological analog of the operation would require signal-passing 
    in both directions to achieve the same effect, which is consistent with observation.
    
    Similarly, a convolutional operation would be expected to have a dense focal point. The ventral V1 region is observed 
    to occupy a smaller region centered around the fovea, 
    compared to the retina-spanning dorsal estuary of the same inputs, consistent with this idea.
    
    We would also expect a more involved routing system to handle convolution via sequencing and spiking 
    rest potential resets. 
    It is indeed observed that the optic nerve stretches out-of-its-way-far from the eye to the back of the head,
    to reach the occipital entrance.
    
    This model reconciles both the what/where-pathway hypothesis, with two disentangled streams,
    and the perceptual/motor-pathway hypothesis, with dorsal-mediated visuo-attention and motor control.
    
    The improved performance is a strong argument for the model. We think the concepts and noted advantages of 
    non-locality, cross-attention with skip connections, and representational disentanglement defined in deep learning
    are pertinent to neuroscience researchers to understanding the technical purpose of the "what"/"where" dichotomy,
    as opposed to other configurations, and disentanglement of perceptual signals, visuo-attention, 
    and motor function in the dorsal region."""


# We can also efficiently substitute the patched MLPs with patched ViTs.

# The operation is linear w.r.t. the number of patches. The ViTs are confined to their respective patches
# and therefore do not infect the total operation with quadratic complexity.
# The cross attentions between streams have the relational advantages of self-attention-based ViTs, but with linear
# time-complexity.

# For that matter, the residual blocks could also be Vision Transformer layers,
# but this would introduce quadratic complexity.

# Can "max-pool" relation-disentanglement style layer by layer using cross attentions and gumbel softmax for escaping
# local hard-attention optima. Can also try to predict dorsal from ventral cross attend as self-supervision.
# Fully architectural. No data augmentation, perturbing, or masking. Debatable whether it can even be called
# self-supervision or if it's just a really good architecture! (Not even locality embeddings)
# Architectural form of this kind of:
# https://medium.com/syncedreview/
# a-leap-forward-in-computer-vision-facebook-ai-says-masked-autoencoders-are-scalable-vision-32c08fadd41f


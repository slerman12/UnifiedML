import math

from torch import nn

from Blocks.Architectures.Residual import ResidualBlock, Residual
from Blocks.Architectures.MLP import MLP
from Blocks.Architectures.MultiHeadAttention import CrossAttention, SelfAttention


class BioNet(nn.Module):
    """Disentangling "What" And "Where" Pathways In CNNs"""

    def __init__(self, input_shape, out_channels, num_patches, depth=3, output_dim=128):
        super().__init__()
        in_channels = input_shape[0]
        out_channels = out_channels

        assert len(num_patches) == 2
        num_patches = [1, *num_patches]  # Channel dimension

        # For MLP
        patch_dim = math.prod(input_shape) / math.prod(num_patches)

        mlp = lambda **args: nn.Sequential(nn.Flatten(-3), MLP(**args))
        mlp_args = dict(out_dim=out_channels, hidden_dim=out_channels, depth=3)

        self.ventral_stream = [ResidualBlock(in_channels, out_channels)]
        self.dorsal_stream = [Patched(mlp, num_patches, in_dim=patch_dim, **mlp_args)]

        for _ in range(depth - 1):
            self.ventral_stream.append(ResidualBlock(out_channels, out_channels))
            self.dorsal_stream.append(Patched(mlp, num_patches, in_dim=out_channels, **mlp_args))

        self.cross_talk = [CrossAttention(out_channels, heads=8) for _ in range(depth)]

        self.attn = SelfAttention(out_channels, heads=8)
        self.pool = nn.AdaptiveAvgPool2d(output_dim ** 0.5)

    def forward(self, input):
        ventral = dorsal = input
        for what, where, talk in zip(self.ventral_stream,
                                     self.dorsal_stream,
                                     self.cross_talk):
            ventral = what(ventral)
            dorsal = talk(where(dorsal), ventral.transpose(1, -1))

        out = self.attn(dorsal).mean(-1)
        # out = self.attn(dorsal).flatten(4).mean(-1)  # for vits
        out = self.pool(out).flatten(1)
        return out

    """Aside from the way the modules are put together (via two disentangled streams), 
    there is only one part of this that we have not seen before, and that is the /Patched/ MLP. 
    
    Besides that, we have a ResidualBlock [resnet], CrossAttention [perceiver], 
    and SelfAttention [all you need].
    
    Here is a schematic visualization for those unfamiliar with Pytorch:
    
    [eye,, self attend + average <- resblocks,: patched mlps <- input
    
    Now, let's go over the Patched MLP. 
    Note that to make this extendable to arbitrary grid/set/sequence sizes,
    we could use a Patched CNN with locality coords or even a Patched SelfAttention, all generalities preserved.
    
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
    
    We would also expect a more involved routing system to handle sequential convolution and spike 
    rest potential resets. 
    It is indeed observed that the optic nerve stretches out-of-its-way-far from the eye to the back of the head,
    to reach the occipital entrance.
    
    This model reconciles both the what/where-pathway hypothesis, with two disentangled streams,
    and the perceptual/motor-pathway hypothesis, with dorsal-mediated visuo-attention-based motor control.
    
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


# Unique instance of a module applied per patch
class Patched(nn.Module):
    def __init__(self, module, num_patches, **module_args):
        super().__init__()
        self.num_patches = num_patches
        self.modules = nn.ModuleList([module(**module_args) for _ in range(math.prod(num_patches))])

    def forward(self, x):
        shape = x.shape[:-len(self.num_patches)]
        tail_shape = x.shape[-len(self.num_patches):]
        patch_shape = [whole / part for whole, part in zip(tail_shape, self.num_patches)]
        x = x.view(-1, *self.num_patches, *patch_shape)
        out = torch.stack([module(patch) for patch, module in zip(x, self.modules)])
        return out.view(*shape, *out.shape[len(shape):])
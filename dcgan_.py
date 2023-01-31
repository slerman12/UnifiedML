from __future__ import print_function
import random
from pathlib import Path

import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
# import tensorflow as tf

# Set random seed for reproducibility
manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)




# Root directory for dataset
dataroot = "Datasets/ReplayBuffer/Classify/CelebA_Train/"


# def load_dataset(split):
#     train_list_ds = tf.data.Dataset.from_tensor_slices(np.load(dataroot.format(split)))
#     train_ds = train_list_ds.map(lambda x: (x, x))
#     return train_ds
#
#
# train_ds = load_dataset('train')
# val_ds = load_dataset('val')
# test_ds = load_dataset('test')

# Number of workers for dataloader
workers = 2

# Batch size during training
batch_size = 128

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 64

# Number of channels in the training images. For color images this is 3
nc = 3

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64

# Number of training epochs
num_epochs = 5

# Learning rate for optimizers
lr = 0.0002

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1




# We can use an image folder dataset the way we have it setup.
# Create the dataset
# dataset = dset.ImageFolder(root=dataroot,
#                            transform=transforms.Compose([
#                                transforms.Resize(image_size),
#                                transforms.CenterCrop(image_size),
#                                transforms.ToTensor(),
#                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#                            ]))
dataset = torchvision.datasets.celeba.CelebA(root=dataroot,
                                             download=True,
                                             transform=transforms.Compose([
                                                 transforms.Resize(image_size),
                                                 transforms.CenterCrop(image_size),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                             ]))


# Create the dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# Plot some training images
real_batch = next(iter(dataloader))
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))


from Blocks.Architectures.Vision.CNN import cnn_broadcast
import Utils

class Generator(nn.Module):
    def __init__(self, input_shape, hidden_dim=64, output_shape=None):
        super().__init__()

        self.input_shape, self.output_shape = Utils.to_tuple(input_shape), Utils.to_tuple(output_shape)
        # Proprioceptive is channel dim
        self.input_shape = tuple(self.input_shape) + (1,) * (3 - len(self.input_shape))  # Broadcast input to 2D

        in_channels = self.input_shape[0]
        out_channels = in_channels if self.output_shape is None else self.output_shape[0]

        self.Generator = nn.Sequential(
            # (hidden_dim * 8) x 4 x 4
            nn.ConvTranspose2d(in_channels, hidden_dim * 8, 4, bias=False),
            nn.BatchNorm2d(hidden_dim * 8),
            nn.ReLU(inplace=True),

            # (hidden_dim * 4) x 8 x 8
            nn.ConvTranspose2d(hidden_dim * 8, hidden_dim * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.ReLU(inplace=True),

            # (hidden_dim * 2) x 16 x 16
            nn.ConvTranspose2d(hidden_dim * 4, hidden_dim * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.ReLU(inplace=True),

            # hidden_dim x 32 x 32
            nn.ConvTranspose2d(hidden_dim * 2, hidden_dim, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),

            # out_channels x 64 x 64
            nn.ConvTranspose2d(hidden_dim, out_channels, 4, 2, 1, bias=False),
            # nn.Identity() if self.output_shape is None else nn.AdaptiveAvgPool2d(self.output_shape[1:])  # Adapts scale
            nn.Tanh()  # TODO added here
        )

        self.apply(weight_init)

    def repr_shape(self, *_):
        return Utils.repr_shape(_, self.Generator)  # cnn_feature_shape doesn't support pre-computing ConvTranspose2d

    def forward(self, *x):
        # Concatenate inputs along channels assuming dimensions allow, broadcast across many possibilities
        lead_shape, x = cnn_broadcast(self.input_shape, x)

        x = self.Generator(x)

        # Restore leading dims
        out = x.view(*lead_shape, *(self.output_shape or x.shape[1:]))
        return out

# TODO uncommnet the adaptive pools
class Discriminator(nn.Module):
    def __init__(self, input_shape, hidden_dim=64, output_shape=None):
        super().__init__()

        self.input_shape, self.output_shape = Utils.to_tuple(input_shape), Utils.to_tuple(output_shape)

        in_channels = self.input_shape[0]

        self.Discriminator = nn.Sequential(
            # hidden_dim x 32 x 32
            # nn.AdaptiveAvgPool2d(64),  # Adapts from different scales  TODO Does averaging cause weird distortion?
            nn.Conv2d(in_channels, hidden_dim, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # (hidden_dim * 2) x 16 x 16
            nn.Conv2d(hidden_dim, hidden_dim * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # (hidden_dim * 4) x 8 x 8
            nn.Conv2d(hidden_dim * 2, hidden_dim * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # (hidden_dim * 8) x 4 x 4
            nn.Conv2d(hidden_dim * 4, hidden_dim * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_dim * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # 1 x 1 x 1
            nn.Conv2d(hidden_dim * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()  # TODO Using Sigmoid with MSE breaks for some reason!
        )

        self.apply(weight_init)

    def repr_shape(self, *_):
        return Utils.cnn_feature_shape(_, self.Discriminator)

    def forward(self, *x):
        # Concatenate inputs along channels assuming dimensions allow, broadcast across many possibilities
        lead_shape, x = cnn_broadcast(self.input_shape, x)

        x = self.Discriminator(x)

        # Restore leading dims
        out = x.view(*lead_shape, *(self.output_shape or x.shape[1:]))
        return out


# TODO perhaps try torch.amax(x, dim=(2,3)) # Global maximum pooling
# TODO Perhaps init agent with .to(memory_format=torch.channels_last) and .half()

# TODO Augs, Compose

# def make_random_square_masks(inputs, mask_size):
#     ##### TODO: Double check that this properly covers the whole range of values. :'( :')
#     if mask_size == 0:
#         return None # no need to cutout or do anything like that since the patch_size is set to 0
#     is_even = int(mask_size % 2 == 0)
#     in_shape = inputs.shape
#
#     # seed centers of squares to cutout boxes from, in one dimension each
#     mask_center_y = torch.empty(in_shape[0], dtype=torch.long, device=inputs.device).random_(mask_size//2-is_even, in_shape[-2]-mask_size//2-is_even)
#     mask_center_x = torch.empty(in_shape[0], dtype=torch.long, device=inputs.device).random_(mask_size//2-is_even, in_shape[-1]-mask_size//2-is_even)
#
#     # measure distance, using the center as a reference point
#     to_mask_y_dists = torch.arange(in_shape[-2], device=inputs.device).view(1, 1, in_shape[-2], 1) - mask_center_y.view(-1, 1, 1, 1)
#     to_mask_x_dists = torch.arange(in_shape[-1], device=inputs.device).view(1, 1, 1, in_shape[-1]) - mask_center_x.view(-1, 1, 1, 1)
#
#     to_mask_y = (to_mask_y_dists >= (-(mask_size // 2) + is_even)) * (to_mask_y_dists <= mask_size // 2)
#     to_mask_x = (to_mask_x_dists >= (-(mask_size // 2) + is_even)) * (to_mask_x_dists <= mask_size // 2)
#
#     final_mask = to_mask_y * to_mask_x ## Turn (y by 1) and (x by 1) boolean masks into (y by x) masks through multiplication. Their intersection is square, hurray! :D
#
#     return final_mask
#
# def batch_cutout(inputs, patch_size):
#     with torch.no_grad():
#         cutout_batch_mask = make_random_square_masks(inputs, patch_size)
#         if cutout_batch_mask is None:
#             return inputs # if the mask is None, then that's because the patch size was set to 0 and we will not be using cutout today.
#         # TODO: Could be fused with the crop operation for sheer speeeeeds. :D <3 :))))
#         cutout_batch = torch.where(cutout_batch_mask, torch.zeros_like(inputs), inputs)
#         return cutout_batch
#
# def batch_crop(inputs, crop_size):
#     with torch.no_grad():
#         crop_mask_batch = make_random_square_masks(inputs, crop_size)
#         cropped_batch = torch.masked_select(inputs, crop_mask_batch).view(inputs.shape[0], inputs.shape[1], crop_size, crop_size)
#         return cropped_batch
#
# def batch_flip_lr(batch_images, flip_chance=.5):
#     with torch.no_grad():
#         # TODO: Is there a more elegant way to do this? :') :'((((
#         return torch.where(torch.rand_like(batch_images[:, 0, 0, 0].view(-1, 1, 1, 1)) < flip_chance, torch.flip(batch_images, (-1,)), batch_images)


# Initializes model weights a la normal
def weight_init(m):
    if isinstance(m, (nn.Conv2d, nn.Conv1d)) or isinstance(m, (nn.ConvTranspose2d, nn.ConvTranspose1d)):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

netG = Generator((100,), ngf, (3, 64, 64)).to(device)

# Create the Discriminator
netD = Discriminator((3, 64, 64), ngf, (1,)).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.

# Print the model
print(netD)





# Initialize BCELoss function
# criterion = nn.MSELoss()
criterion = nn.BCELoss()

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_noise = torch.randn(64, nz, 1, 1, device=device)

# Establish convention for real and fake labels during training
real_label = 1.
fake_label = 0.

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))





# Training Loop

# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
iters = 0

print("Starting Training Loop...")
# For each epoch
for epoch in range(num_epochs):
    # For each batch in the dataloader
    for i, data in enumerate(dataloader, 0):

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        netD.zero_grad()
        # Format batch
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        # Forward pass real batch through D
        output = netD(real_cpu).view(-1)
        # Calculate loss on all-real batch
        errD_real = criterion(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        # Generate fake image batch with G
        fake = netG(noise)
        label.fill_(fake_label)
        # Classify all fake batch with D
        output = netD(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label)
        # Calculate the gradients for this batch, accumulated (summed) with previous gradients
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Compute error of D as sum over the fake and the real batches
        errD = errD_real + errD_fake
        # Update D
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(fake).view(-1)
        # Calculate G's loss based on this output
        # errG = criterion(output, label)  # TODO MSE better than nothing because diminishes gradients closer to 0, 1
        errG = -output.log().mean()  # TODO Try
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()

        # Output training stats
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

        iters += 1






plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()




# Grab a batch of real images from the dataloader
real_batch = next(iter(dataloader))

# Plot the real images
plt.figure(figsize=(15,15))
plt.subplot(1,2,1)
plt.axis("off")
plt.title("Real Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

# Plot the fake images from the last epoch
plt.subplot(1,2,2)
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(img_list[-1],(1,2,0)))
plt.show()
path = Path('./Benchmarking/DCGAN/AC2Agent/classify/CelebA_1_Video_Image')
path.mkdir(parents=True, exist_ok=True)
plt.savefig(path / 'generated.png')

plt.close()

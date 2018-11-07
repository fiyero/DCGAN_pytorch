import torchvision
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import matplotlib
import random

t1=time.time()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Setting some hyperparameters
batch_size = 128
image_size = 64
nz=100
nc=3
num_epochs=5
ngf=64
ndf=64
workers=2
path='E:\Projects\dcgan\celeba'
beta1 = 0.5
lr_d=0.0005
lr_g=0.0005

# Loading the dataset
dataset = datasets.ImageFolder(root=path,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))

dataloader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle = True, num_workers =workers)

def weights_init(m):

    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class G(nn.Module):

    def __init__(self):
        super(G, self).__init__()
        self.main = nn.Sequential(

            nn.ConvTranspose2d(nz, ngf*8, 4, 1, 0, bias = False),
            nn.BatchNorm2d(ngf*8),

            nn.ReLU(True),
            nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias = False),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias = False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias = False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias = False),
            nn.Tanh()
        )

    def forward(self, input):
        output = self.main(input)
        return output

# instantiate the generator netG
netG = G()
netG=nn.DataParallel(netG)
netG.to(device)
netG.apply(weights_init)


# Defining the discriminator
class D(nn.Module):

    def __init__(self):
        super(D, self).__init__()
        self.main = nn.Sequential(

            nn.Conv2d(nc, ndf, 4, 2, 1, bias = False),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias = False),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace = True),

            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias = False),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Conv2d(ndf*4, ndf*8, 4, 2, 1, bias = False),
            nn.BatchNorm2d(ndf*8),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Conv2d(ndf*8, 1, 4, 1, 0, bias = False),
            nn.Sigmoid()

        )

    def forward(self, input):
        output = self.main(input)

        return output.view(-1)

criterion = nn.BCELoss()
optimizerD = optim.Adam(netD.parameters(), lr=lr_d, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr_g, betas=(beta1, 0.999))
G_losses=[]
D_losses=[]
img_list=[]

fixed_noise=torch.randn(64,nz,1,1).to(device)
real_label = 1
fake_label = 0

for epoch in range(num_epochs):

    for i, data in enumerate(dataloader, 0):
        netD.zero_grad()
        real_img, _ = data
        input_real_img_tensor= real_img.to(device)
        b_size=input_real_img_tensor.size()[0]
        target=torch.full((b_size,), real_label).to(device)
        output = netD(input_real_img_tensor)
        D_x= output.mean().item()
        errD_real = criterion(output, target)

        noise = torch.randn(input_real_img_tensor.size()[0], nz, 1, 1).to(device)
        fake_img_tensor = netG(noise)
        target=torch.full((b_size,), fake_label).to(device)
        output = netD(fake_img_tensor.detach())
        D_G_z1= output.mean().item()
        errD_fake = criterion(output, target)
        errD = errD_real + errD_fake
        errD.backward()
        optimizerD.step()


        netG.zero_grad()
        target=torch.full((b_size,), real_label).to(device)
        output = netD(fake_img_tensor)
        D_G_z2= output.mean().item()
        errG = criterion(output, target)
        errG.backward()
        optimizerG.step()

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())
        if i % 100 == 0:
            print(f'[{epoch+1}/{num_epochs}][{i}/{len(dataloader)}] Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f}\tD(x): {D_x:.4f}\tD(G(z)): {D_G_z1:.4f} / {D_G_z2:.4f}')
            vutils.save_image(real_img, '%s/real_img%03d.jpg' % ("./celeba/load_weight_progress2", epoch+1), normalize = True)

            with torch.no_grad():
                fake = netG(noise).detach()
                vutils.save_image(fake.data, '%s/fake_img_epoch_%03d.jpg' % ("./celeba/load_weight_progress", epoch+1), normalize = True)
            with torch.no_grad():
                fake=netG(fixed_noise).detach()
                vutils.save_image(fake.data, '%s/fake_img_version_epoch_%03d.jpg' % ("./celeba/load_weight_results2", epoch+1), normalize = True)

                img_list.append(vutils.make_grid(fake,padding=2,normalize=True))

plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()

real_batch = next(iter(dataloader))
plt.figure(figsize=(24,24))
plt.subplot(1,2,1)
plt.axis("off")
plt.title("Real Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0], padding=2, normalize=True),(1,2,0)))


plt.figure(figsize=(24,24))
plt.subplot(1,2,2)
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(img_list[-1],(1,2,0)))
plt.show()

t2=time.time()
print(f'it take {t2-t1}s to complete')

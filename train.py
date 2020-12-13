import torch
import torch.nn as nn
import torchvision
from torchvision.datasets import MNIST
from torch.optim import Adam
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

from gan import Generator, Discriminator
from tqdm import tqdm
import datetime



def train(discriminator, 
          generator, 
          optimizer,
          dataloader, 
          loss_fn, 
          device, 
          writer, 
          epochs=1000):
    """
    Training loop.
    """
    optimizer_g = optimizer(generator.parameters(), lr=0.0001)
    optimizer_d = optimizer(discriminator.parameters(), lr=0.0001)
    

    total_it = 0
    for epoch in range(epochs):
        iterator_bar = tqdm(dataloader)
        for sample, _ in iterator_bar:

            real_samples = sample.to(device)

            real = torch.Tensor([0]).to(device).float()
            fake = torch.Tensor([1]).to(device).float()
            real = real.repeat(sample.shape[0])
            fake = fake.repeat(sample.shape[0])


            latent = torch.randn(sample.shape[0], 128).to(device)
            generator_samples = generator(latent)
            # train discriminator - real
            optimizer_d.zero_grad()
            class_probs_dr = discriminator(real_samples)
            loss_dr = loss_fn(class_probs_dr, real)

            # train discriminator - fake
            class_probs_df = discriminator(generator_samples)
            loss_df = loss_fn(class_probs_df, fake)
            loss_d = (loss_df+loss_dr)/2
            loss_d.backward()
            optimizer_d.step()

            # train generator
            loss_g_avg = 0
            for i in range(5):
                latent = torch.normal(0, 1, (sample.shape[0], 128)).to(device)
                generator_samples = generator(latent)
                optimizer_g.zero_grad()
                class_probs_g = discriminator(generator_samples)
                loss_g = loss_fn(class_probs_g, real)
                loss_g.backward()
                loss_g_avg += loss_g
                optimizer_g.step()
            loss_g_avg /= 10

            # loss
            iterator_bar.set_description('Epoch {} - G Loss: {} DR Loss: {} DF Loss: {}'.format(epoch, loss_g, loss_dr, loss_df))
            iterator_bar.refresh()
            
            generator_grid = torchvision.utils.make_grid(generator_samples)
            input_grid = torchvision.utils.make_grid(real_samples)
            writer.add_image('Generated Images', generator_grid*0.5+0.5, total_it)
            writer.add_image('Input Images', input_grid*0.5+0.5, total_it)
            writer.add_scalar('Generator Loss', loss_g_avg, total_it)
            writer.add_scalar('Discriminator Loss Real', loss_dr, total_it)
            writer.add_scalar('Discriminator Loss Fake', loss_df, total_it)

            total_it += 1

    return generator


if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print("Training starting on device {}".format(device))
    writer = SummaryWriter('/blue/vemuri/josebouza/projects/DCGAN/logs/runs/{}'.format(datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")))
    tr = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize(mean=(0.5,), std=(0.5,))])
    dataset = MNIST('/blue/vemuri/josebouza/data/mnist/', 
                    download=True,
                    transform=tr)
    dataloader = DataLoader(dataset, 
                            batch_size=128,
                            num_workers=4)
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    train(discriminator, generator, torch.optim.Adam, dataloader, nn.BCEWithLogitsLoss().to(device), device, writer)
    torch.save(model.state_dict(), '/blue/vemuri/josebouza/projects/GAN/model.pt')

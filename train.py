import torch
from torch import nn
from processing.data_preprocessing import *
from model.model import *
import os
num_epochs = 10
autoencoder = Autoencoder()

device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")
autoencoder = autoencoder.to(device)

num_params = sum(p.numel() for p in autoencoder.parameters() if p.requires_grad)
print('Number of parameters: %d' % num_params)

optimizer = torch.optim.Adam(params=autoencoder.parameters(), lr=learning_rate, weight_decay=1e-5)
loss = nn.MSELoss()

# Set the autoencoder in evaluation mode
autoencoder.train()

train_loss_avg = []

# testloader_iter = iter(trainloader)

# for batch_idx, (data, target) in enumerate(testloader_iter):
#     print(f"Batch {batch_idx}:")
#     print("Data:", data.squeeze())
#     print("Target:", target)

print('Training ...')

checkpoint_interval = 1

if os.path.exists('model_checkpoint.pt'):
    checkpoint = torch.load('model_checkpoint.pt')
    autoencoder.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    initial_epoch = checkpoint['epoch']
    print("model incarcat din checkpoint")

for epoch in range(num_epochs):
    train_loss_avg.append(0)
    num_batches = 0
    print('--------')
    for i, _ in trainnoisyloader:
        train_batch, _ = next(iter(trainloader))
        train_noisy_batch, _ = next(iter(trainnoisyloader))
        print(train_batch.size())

        train_batch = train_batch.to(device)
        train_noisy_batch = train_noisy_batch.to(device)
        if train_batch is None:
            break  

        image_batch_recon = autoencoder(train_noisy_batch)
        print(image_batch_recon.size())
        
        # Reconstruction loss
        reconstruction_loss = loss(image_batch_recon, train_batch)

        optimizer.zero_grad()
        reconstruction_loss.backward()
        optimizer.step()

        if epoch % checkpoint_interval == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': autoencoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, 'model_checkpoint.pt')

        train_loss_avg[-1] += reconstruction_loss.item()
        num_batches += 1

    train_loss_avg[-1] /= num_batches
    print('Epoch [%d / %d] Average Reconstruction Loss: %f' % (epoch+1, num_epochs, train_loss_avg[-1]))



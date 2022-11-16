

if __name__ == '__main__':

    import os
    import urllib.request
    from urllib.error import HTTPError
    import sys
    import matplotlib.pyplot as plt
    import pytorch_lightning as pl
    import torch
    import tensorboard as tb
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    import torch.utils.data as data
    import torchvision
    from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter('/Users/JasperHilliard/Documents/Git testing Project/saved_models/tutorial9')
    from torchvision import transforms

    from torchvision.datasets import CIFAR10
    from tqdm.notebook import tqdm
    import torch.multiprocessing as mp
    import Bigmodel
    import torch
    import torchvision
    from torch import nn
    from torch import optim
    from torch import tensor
    import torch.utils.data as data
    import torchvision.datasets as datasets
    import torchvision.transforms as transforms
    import torch.nn.functional as F
    import pytorch_lightning as pl
    import matplotlib.pyplot as plt
    from torchvision.datasets import CIFAR10


    DATASET_PATH = '/Users/JasperHilliard/Documents/Git testing Project/ganang'
    CHECKPOINT_PATH = os.environ.get("/Users/JasperHilliard/Documents/Git testing Project", "saved_models/tutorial9")



    DATASET_PATH = '/Users/JasperHilliard/Documents/Git testing Project/ganang'
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    train_dataset = CIFAR10(root=DATASET_PATH, train=True, transform=transform, download=True)
    pl.seed_everything(69)
    trainset, val_set = data.random_split(train_dataset, [45000, 5000])

    # Loading the test setw
    test_set = CIFAR10(root=DATASET_PATH, train=False, transform=transform, download=True)


    trainloader = data.DataLoader(trainset, batch_size=128, shuffle=True, drop_last=True, pin_memory=True,
                                  num_workers=4)
    val_loader = data.DataLoader(val_set, batch_size=128, shuffle=False, drop_last=False, num_workers=4)
    test_loader = data.DataLoader(test_set, batch_size=128, shuffle=False, drop_last=False, num_workers=4)

    class Encoder(nn.Module):
        def __init__(self, num_input_channels: int, base_channel_size: int, latent_dim: int, act_fn: object = nn.ReLU):
            super().__init__()
            print('latent dim:', latent_dim)
            channel = num_input_channels
            double_channel = channel * 2
            print('initially has the encoder init initialized.')
            self.net = nn.Sequential(
                nn.Conv2d(num_input_channels, channel, kernel_size=3, padding=1, stride=2),  # 32x32 => 16x16
                act_fn(),
                nn.Conv2d(channel, channel, kernel_size=3, padding=1),
                act_fn(),
                nn.Conv2d(channel, double_channel, kernel_size=3, padding=1, stride=2),  # 16x16 => 8x8
                act_fn(),
                nn.Conv2d(2 * channel, double_channel, kernel_size=3, padding=1),
                act_fn(),
                nn.Conv2d(2 * channel, double_channel, kernel_size=3, padding=1, stride=2),  # 8x8 => 4x4
                act_fn(),
                nn.Flatten(),
                nn.Linear(2 * 16 * channel, latent_dim),
            )
        print('defining forward')
        def forward(self, x):
            print('going forward')
            return self.net(x)


    class Decoder(nn.Module):
            def __init__(self, num_input_channels: int, base_channel_size: int, latent_dim: int,
                         act_fn: object = nn.ReLU):
                super().__init__()
                channel = base_channel_size
                self.linear = nn.Sequential(nn.Linear(latent_dim, 2 * 16 * channel), act_fn())
                print('initially has the decoder init initialized.')
                self.net = nn.Sequential(
                    nn.ConvTranspose2d(
                        2 * channel, 2 * channel, kernel_size=3, output_padding=1, padding=1, stride=2
                    ),
                    act_fn(),
                    nn.ConvTranspose2d(2 * channel, 2 * channel, kernel_size=3, padding=1),
                    act_fn(),
                    nn.ConvTranspose2d(2 * channel, channel, kernel_size=3, output_padding=1, padding=1, stride=2),
                    # 8x8 => 16x16
                    act_fn(),
                    nn.ConvTranspose2d(channel, channel, kernel_size=3, padding=1),
                    act_fn(),
                    nn.ConvTranspose2d(
                        channel, num_input_channels, kernel_size=3, output_padding=1, padding=1, stride=2
                    ),
                    nn.Tanh(),
                )

            def forward(self, x):
                x = self.linear(x)
                x = x.reshape(x.shape[0], -1, 4, 4)
                x = self.net(x)
                return x


    class Autoencoder(pl.LightningModule):
            def __init__(
                    self,
                    base_channel_size: int,
                    latent_dim: int,
                    encoder_class: object = Encoder,
                    decoder_class: object = Decoder,
                    num_input_channels: int = 3,
                    width: int = 32,
                    height: int = 32,
            ):
                super().__init__()
                self.save_hyperparameters()
                # Creating encoder and decoder
                self.encoder = encoder_class(num_input_channels, base_channel_size, latent_dim)
                self.decoder = decoder_class(num_input_channels, base_channel_size, latent_dim)

            def forward(self, x):
                print('forward function defined')
                z = self.encoder(x)
                print('encoder forward pass')
                x_hat = self.decoder(z)
                print('decoder forward done')
                return x_hat

            def _get_reconstruction_loss(self, batch):
                """Given a batch of images, this function returns the reconstruction loss (MSE in our case)"""
                x, _ = batch  # We do not need the labels
                x_hat = self.forward(x)
                loss = F.mse_loss(x, x_hat, reduction="none")
                loss = loss.sum(dim=[1, 2, 3]).mean(dim=[0])
                return loss

            def configure_optimizers(self):
                optimizer = optim.Adam(self.parameters(), lr=2e-3)
                # Using a scheduler is optional but can be helpful.
                # The scheduler reduces the LR if the validation performance hasn't improved for the last N epochs
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.2, patience=20,
                                                                 min_lr=5e-5)
                return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

            def training_step(self, batch, batch_idx):

                loss = self._get_reconstruction_loss(batch)
                loss = self._get_reconstruction_loss(batch)
                self.log("train_loss", loss)

                return loss

            def validation_step(self, batch, batch_idx):

                loss = self._get_reconstruction_loss(batch)
                self.log("val_loss", loss)

            def test_step(self, batch, batch_idx):
                loss = self._get_reconstruction_loss(batch)
                self.log("test_loss", loss)


    class GenerateCallback(pl.Callback):
        def __init__(self, input_imgs, every_n_epochs=1):
            super().__init__()
            self.input_imgs = input_imgs  # Images to reconstruct during training
            # Only save those images every N epochs (otherwise tensorboard gets quite large)
            self.every_n_epochs = every_n_epochs

        def on_epoch_end(self, trainer, pl_module):
            if trainer.current_epoch % self.every_n_epochs == 0:
                # Reconstruct images
                input_imgs = self.input_imgs.to(pl_module.device)
                with torch.no_grad():
                    pl_module.eval()
                    reconst_imgs = pl_module(input_imgs)
                    pl_module.train()
                # Plot and add to tensorboard
                imgs = torch.stack([input_imgs, reconst_imgs], dim=1).flatten(0, 1)
                grid = torchvision.utils.make_grid(imgs, nrow=2, normalize=True, range=(-1, 1))
                writer.add_image('cifar_images', grid)
                writer.close()
                print('tensorboard uploading done')

                trainer.logger.experiment.add_image("Reconstructions", grid, global_step=trainer.global_step)
                sys.exit()

    def get_train_images(num):
        return torch.stack([train_dataset[i][0] for i in range(num)], dim=0)
    def train_cifar(latent_dim):
        # Create a PyTorch Lightning trainer with the generation callback
        trainer = pl.Trainer(
            default_root_dir=os.path.join(CHECKPOINT_PATH, "cifar10_%i" % latent_dim),
            gpus=0,
            max_epochs=12,
            callbacks=[
                ModelCheckpoint(save_weights_only=True),
                GenerateCallback(get_train_images(8), every_n_epochs=10),
                LearningRateMonitor("epoch"),
            ],
        )
        trainer.logger._log_graph = True  # If True, we plot the computation graph in tensorboard
        trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

        # Check whether pretrained model exists. If yes, load it and skip training
        pretrained_filename = os.path.join(CHECKPOINT_PATH, "cifar10_%i.ckpt" % latent_dim)
        if os.path.isfile(pretrained_filename):
            print("Found pretrained model, loading...")
            model = Autoencoder.load_from_checkpoint(pretrained_filename)
        else:
            model = Autoencoder(base_channel_size=32, latent_dim=latent_dim)
            trainer.fit(model, trainloader, val_loader)
        # Test best model on validation and test set
        val_result = trainer.test(model, dataloaders=val_loader, verbose=False)
        test_result = trainer.test(model, dataloaders=test_loader, verbose=False)
        result = {"test": test_result, "val": val_result}
        return model, result

    model_dict = {}
    for latent_dim in [128, 256, 384]:
        model_ld, result_ld = train_cifar(latent_dim)
        model_dict[latent_dim] = {"model": model_ld, "result": result_ld}



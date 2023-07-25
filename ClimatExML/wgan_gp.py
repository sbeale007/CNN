import pytorch_lightning as pl
import torch.nn.functional as F
import torch
from ClimatExML.models import Generator
from ClimatExML.mlflow_tools.mlflow_tools import (
    gen_grid_images,
    log_metrics_every_n_steps,
    log_pytorch_model,
)
from ClimatExML.loader import ClimatExMLLoader
from ClimatExML.losses import content_loss, SSIM_Loss
from torch.utils.data import DataLoader
from torchmetrics.functional import (
    mean_absolute_error,
    mean_squared_error,
    multiscale_structural_similarity_index_measure,
)
import mlflow
import matplotlib.pyplot as plt


class SuperResolutionWGANGP(pl.LightningModule):
    def __init__(
        self,
        batch_size: int = 16,
        num_workers: int = 24,
        learning_rate: float = 0.00025,
        b1: float = 0.9,
        b2: float = 0.999,
        gp_lambda: float = 10,
        alpha: float = 1e-3,
        lr_shape: tuple = (3, 64, 64),
        hr_shape: tuple = (2, 512, 512),
        n_critic: int = 5,
        log_every_n_steps: int = 100,
        artifact_path: str = None,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        # data
        self.num_workers = num_workers

        # training
        self.learning_rate = learning_rate
        self.b1 = b1
        self.b2 = b2
        self.batch_size = batch_size
        self.lr_shape = lr_shape
        self.hr_shape = hr_shape
        self.gp_lambda = gp_lambda
        self.n_critic = n_critic
        self.alpha = alpha
        self.log_every_n_steps = log_every_n_steps
        self.artifact_path = artifact_path

        # networks
        n_covariates, lr_dim, _ = self.lr_shape
        n_predictands, hr_dim, _ = self.hr_shape
        # DEBUG coarse_dim_n, fine_dim_n, n_covariates, n_predictands
        self.G = Generator(lr_dim, hr_dim, n_covariates, n_predictands)

        self.automatic_optimization = False

        # self.register_buffer("gp_alpha", torch.rand(current_batch_size, 1, 1, 1, requires_grad=True).expand_as(real_samples))
        # self.register_buffer("gp_ones", torch.ones(critic_interpolated.size(), requires_grad=True))

    def compute_gradient_penalty(self, real_samples, fake_samples):
        """Calculates the gradient penalty loss for WGAN GP"""
        current_batch_size = real_samples.size(0)
        # Calculate interpolation

        # gradient penalty
        gp_alpha = (
            torch.rand(current_batch_size, 1, 1, 1, requires_grad=True)
            .expand_as(real_samples)
            .to(real_samples)
        )

        interpolated = gp_alpha * real_samples.data + (1 - gp_alpha) * fake_samples.data

        # Calculate probability of interpolated examples
        critic_interpolated = self.C(interpolated)

        # self.register_buffer("gp_ones", torch.ones(critic_interpolated.size(), requires_grad=True))

        # Calculate gradients of probabilities with respect to examples
        gradients = torch.autograd.grad(
            outputs=critic_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones(critic_interpolated.size(), requires_grad=True).to(
                real_samples
            ),
            create_graph=True,
            retain_graph=True,
        )[0]

        # Gradients have shape (batch_size, num_channels, img_width, img_height),
        # so flatten to easily take norm per example in batch
        gradients = gradients.view(current_batch_size, -1)

        # Derivatives of the gradient close to 0 can cause problems because of
        # the square root, so manually calculate norm and add epsilon
        gradients_norm = torch.sqrt(torch.sum(gradients**2, dim=1) + 1e-12)

        # Return gradient penalty
        return self.gp_lambda * ((gradients_norm - 1) ** 2).mean()

    def training_step(self, batch, batch_idx):
        # train generator
        lr, hr = batch[0]
        g_opt = self.optimizers()
        # sr = self.G(lr).detach()
        # gradient_penalty = self.compute_gradient_penalty(hr, sr)

        self.toggle_optimizer(g_opt)
        sr = self.G(lr)
        loss_g = content_loss(
            sr, hr
        )

        # self.go_downhill(g_opt, loss_g)
        self.manual_backward(loss_g)
        g_opt.step()
        g_opt.zero_grad(set_to_none=True)
        self.untoggle_optimizer(g_opt)

        loss_g.detach()
        sr.detach()
        hr.detach()
        lr.detach()


        self.log_dict(
            {
                "MAE": content_loss(sr, hr),
                "MSE": mean_squared_error(sr, hr),
                "MSSIM": SSIM_Loss(sr, hr, any),
                "loss": loss_g,
                # "Wasserstein Distance": torch.mean(self.C(hr))
                # - torch.mean(self.C(sr)),
            }
        )

        if (batch_idx + 1) % self.log_every_n_steps == 0:
            fig = plt.figure(figsize=(30, 10))
            for var in range(lr.shape[1]):
                self.logger.experiment.log_figure(
                    mlflow.active_run().info.run_id,
                    gen_grid_images(
                        var,
                        fig,
                        self.G,
                        lr,
                        hr,
                        self.batch_size,
                        n_examples=3,
                        cmap="viridis",
                    ),
                    f"train_images_{var}_{self.current_epoch}_{batch_idx + 1}.png",
                )
                plt.close()

    def go_downhill(self, opt, loss):
        self.manual_backward(loss)
        opt.step()
        opt.zero_grad()

    def test_step(self, batch, batch_idx):
        # if (batch_idx + 1) % self.log_every_n_steps == 0:
        lr, hr = batch
        sr = self.G(lr)
        self.log_dict(
            {
                "Test MAE": content_loss(sr, hr),
                "Test MSE": mean_squared_error(sr, hr),
                "Test MSSIM": SSIM_Loss(sr, hr, any),
            #     "Test Wasserstein Distance": torch.mean(self.C(hr))
            #     - torch.mean(self.C(sr)),
            }
        )

        if (batch_idx + 1) % self.log_every_n_steps == 0:
            fig = plt.figure(figsize=(30, 10))
            for var in range(lr.shape[1]):
                self.logger.experiment.log_figure(
                    mlflow.active_run().info.run_id,
                    gen_grid_images(
                        var,
                        fig,
                        self.G,
                        lr,
                        hr,
                        self.batch_size,
                        n_examples=3,
                        cmap="viridis",
                    ),
                    f"test_images_{var}.png",
                )
                plt.close()

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(
            self.G.parameters(), lr=self.learning_rate, betas=(self.b1, self.b2)
        )
        # opt_d = torch.optim.Adam(
        #     self.C.parameters(), lr=self.learning_rate, betas=(self.b1, self.b2)
        # )
        return opt_g
    
    def forward(self, x):
        x = self.G(x)
        return x

    # def on_train_epoch_end(self) -> None:
    #     log_pytorch_model(self.G, f"{self.artifact_path}/generator")
    #     log_pytorch_model(self.C, f"{self.artifact_path}/critic")

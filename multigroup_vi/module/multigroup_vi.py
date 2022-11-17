import numpy as np
import torch
import torch.nn.functional as F
from scvi import REGISTRY_KEYS
from scvi.distributions import ZeroInflatedNegativeBinomial
from scvi.module.base import BaseModuleClass, LossRecorder, auto_move_data
from scvi.nn import DecoderSCVI, Encoder, one_hot
from torch.distributions import Normal
from torch.distributions import kl_divergence as kl

torch.backends.cudnn.benchmark = True


class MultiGroupVIModule(BaseModuleClass):
    """
    Skeleton Variational auto-encoder model.

    Here we implement a basic version of scVI's underlying VAE [Lopez18]_.
    This implementation is for instructional purposes only.

    Parameters
    ----------
    n_input
        Number of input genes
    library_log_means
        1 x n_batch array of means of the log library sizes. Parameterizes prior on
        library size if not using observed library size.
    library_log_vars
        1 x n_batch array of variances of the log library sizes. Parameterizes prior on
        library size if not using observed library size.
    n_batch
        Number of batches, if 0, no batch correction is performed.
    n_hidden
        Number of nodes per hidden layer
    n_shared_latent
        Dimensionality of the shared latent space.
    n_private_latent
        Dimensionality of the group-specific latent spaces.
    n_layers
        Number of hidden layers used for encoder and decoder NNs
    dropout_rate
        Dropout rate for neural networks
    wasserstein_penalty
        Strength of the wasserstein-distance penalty discouraging shared information
        from being captured by the private latent spaces.
    """

    def __init__(
        self,
        n_input: int,
        library_log_means: np.ndarray,
        library_log_vars: np.ndarray,
        n_groups: int,
        n_batch: int = 0,
        n_hidden: int = 128,
        n_shared_latent: int = 10,
        n_private_latent: int = 10,
        n_layers: int = 1,
        dropout_rate: float = 0.1,
        wasserstein_penalty: float = 1.0,
    ):
        super().__init__()
        self.n_shared_latent = n_shared_latent
        self.n_private_latent = n_private_latent
        self.n_groups = n_groups
        self.n_batch = n_batch
        # this is needed to comply with some requirement of the VAEMixin class
        self.latent_distribution = "normal"
        self.wasserstein_penalty = wasserstein_penalty

        self.register_buffer(
            "library_log_means", torch.from_numpy(library_log_means).float()
        )
        self.register_buffer(
            "library_log_vars", torch.from_numpy(library_log_vars).float()
        )

        self.register_buffer(
            "gammas", torch.FloatTensor([10 ** x for x in range(-6, 7)])
        )

        # setup the parameters of your generative model, as well as your inference model
        self.px_r = torch.nn.Parameter(torch.randn(n_input))
        # z encoder goes from the n_input-dimensional data to an n_latent-d
        # latent space representation. This latent representation is shared among all groups.
        self.z_encoder = Encoder(
            n_input,
            n_shared_latent,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
        )

        # Each t encoder goes from the n_input-dimensional data to an n_private_latent-d
        # latent space representation. Each t encoder is specific to its corresponding group
        # of data.
        self.t_encoders = torch.nn.ModuleList(
            [
                Encoder(
                    n_input,
                    n_private_latent,
                    n_layers=n_layers,
                    n_hidden=n_hidden,
                    dropout_rate=dropout_rate,
                )
                for _ in range(self.n_groups)
            ]
        )

        # l encoder goes from n_input-dimensional data to 1-d library size
        self.l_encoder = Encoder(
            n_input,
            1,
            n_layers=1,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
        )
        # decoder goes from n_latent-dimensional space to n_input-d data
        self.decoder = DecoderSCVI(
            n_shared_latent + n_groups * n_private_latent,
            n_input,
            n_layers=n_layers,
            n_hidden=n_hidden,
        )

    def _get_inference_input(self, tensors_by_group):
        """Parse the dictionary to get appropriate args"""

        x = torch.cat([group[REGISTRY_KEYS.X_KEY] for group in tensors_by_group], dim=0)
        group_labels = torch.cat([group["group"] for group in tensors_by_group], dim=0)
        input_dict = dict(x=x, group_labels=group_labels)
        return input_dict

    def _get_generative_input(self, tensors, inference_outputs):
        z = inference_outputs["z"]
        t_all = inference_outputs["t_all"]
        library = inference_outputs["library"]

        input_dict = {
            "z": z,
            "t_all": t_all,
            "library": library,
        }
        return input_dict

    @auto_move_data
    def inference(self, x, group_labels):
        """
        High level inference method.

        Runs the inference (encoder) model.
        """

        # log the input to the variational distribution for numerical stability
        x_ = torch.log(1 + x)
        # get variational parameters via the encoder networks
        qz_m, qz_v, z = self.z_encoder(x_)
        ql_m, ql_v, library = self.l_encoder(x_)

        qt_m_list = []
        qt_v_list = []
        t_list = []
        for t_encoder in self.t_encoders:
            qt_m, qt_v, t = t_encoder(x)
            qt_m_list.append(qt_m)
            qt_v_list.append(qt_v)
            t_list.append(t)

        qt_m_all = torch.cat(qt_m_list, dim=1)
        qt_v_all = torch.cat(qt_v_list, dim=1)
        t_all = torch.cat(t_list, dim=1)

        latent_space_size_range = torch.arange(self.n_private_latent).to(self.device)
        latent_space_size_range = latent_space_size_range.repeat((t_all.shape[0], 1))
        idxs = latent_space_size_range + (group_labels * self.n_private_latent)

        mask = torch.zeros(t_all.shape).to(self.device)
        mask[torch.arange(mask.size(0)).unsqueeze(1), idxs.long()] = 1
        mask = mask.detach()

        t_all = t_all * mask

        outputs = dict(
            z=z,
            qz_m=qz_m,
            qz_v=qz_v,
            ql_m=ql_m,
            ql_v=ql_v,
            t_all=t_all,
            qt_m_all=qt_m_all,
            qt_v_all=qt_v_all,
            library=library,
            mask=mask,
        )
        return outputs

    @auto_move_data
    def generative(self, z, t_all, library):
        """Runs the generative model."""

        # form the parameters of the ZINB likelihood
        px_scale, _, px_rate, px_dropout = self.decoder(
            "gene", torch.cat([z, t_all], dim=1), library
        )
        px_r = torch.exp(self.px_r)

        return dict(
            px_scale=px_scale, px_r=px_r, px_rate=px_rate, px_dropout=px_dropout
        )

    def loss(
        self,
        tensors_by_group,
        inference_outputs,
        generative_outputs,
        kl_weight: float = 1.0,
    ):
        x = torch.cat([group[REGISTRY_KEYS.X_KEY] for group in tensors_by_group], dim=0)
        qz_m = inference_outputs["qz_m"]
        qz_v = inference_outputs["qz_v"]
        qt_m_all = inference_outputs["qt_m_all"]
        qt_v_all = inference_outputs["qt_v_all"]
        ql_m = inference_outputs["ql_m"]
        ql_v = inference_outputs["ql_v"]
        px_rate = generative_outputs["px_rate"]
        px_r = generative_outputs["px_r"]
        px_dropout = generative_outputs["px_dropout"]
        mask = inference_outputs["mask"]

        mean = torch.zeros_like(qz_m)
        scale = torch.ones_like(qz_v)

        kl_divergence_z = kl(Normal(qz_m, torch.sqrt(qz_v)), Normal(mean, scale)).sum(
            dim=1
        )

        batch_index = torch.cat(
            [group[REGISTRY_KEYS.BATCH_KEY] for group in tensors_by_group], dim=0
        )
        n_batch = self.library_log_means.shape[1]
        local_library_log_means = F.linear(
            one_hot(batch_index, n_batch), self.library_log_means
        )
        local_library_log_vars = F.linear(
            one_hot(batch_index, n_batch), self.library_log_vars
        )

        qt_m_masked = torch.masked_select(qt_m_all, mask > 0).reshape(
            -1, self.n_private_latent
        )
        qt_v_masked = torch.masked_select(qt_v_all, mask > 0).reshape(
            -1, self.n_private_latent
        )

        mean = torch.zeros_like(qt_m_masked)
        scale = torch.ones_like(qt_v_masked)

        kl_divergence_t = kl(
            Normal(qt_m_masked, torch.sqrt(qt_v_masked)), Normal(mean, scale)
        ).sum(dim=1)

        kl_divergence_l = kl(
            Normal(ql_m, torch.sqrt(ql_v)),
            Normal(local_library_log_means, torch.sqrt(local_library_log_vars)),
        ).sum(dim=1)

        qt_tilde_m_masked = torch.masked_select(qt_m_all, mask == 0).reshape(
            -1, self.n_groups - 1, self.n_private_latent
        )
        qt_tilde_v_masked = torch.masked_select(qt_v_all, mask == 0).reshape(
            -1, self.n_groups - 1, self.n_private_latent
        )

        reconst_loss = (
            -ZeroInflatedNegativeBinomial(mu=px_rate, theta=px_r, zi_logits=px_dropout)
            .log_prob(x)
            .sum(dim=-1)
        )

        kl_local_for_warmup = kl_divergence_z + kl_divergence_t
        kl_local_no_warmup = kl_divergence_l

        weighted_kl_local = kl_local_for_warmup + kl_local_no_warmup

        mean = torch.zeros_like(qt_tilde_m_masked)
        scale = torch.ones_like(qt_tilde_v_masked)

        wasserstein_loss = (
            torch.norm(qt_tilde_m_masked, dim=-1) ** 2
            + torch.norm(torch.sqrt(qt_tilde_v_masked), dim=-1) ** 2
        ).sum(dim=-1)

        loss = torch.mean(
            reconst_loss
            + kl_weight
            * (weighted_kl_local + self.wasserstein_penalty * wasserstein_loss)
        )

        kl_local = dict(
            kl_divergence_l=kl_divergence_l,
            kl_divergence_z=kl_divergence_z,
            kl_divergence_t=kl_divergence_t,
        )
        kl_global = torch.tensor(0.0)

        return LossRecorder(
            loss,
            reconst_loss,
            kl_local,
            kl_global,
            wasserstein_loss=wasserstein_loss.sum(),
        )

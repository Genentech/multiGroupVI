import logging
import warnings
from functools import partial
from typing import Dict, Iterable, List, Literal, Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch
from anndata import AnnData
from scvi import REGISTRY_KEYS
from scvi.data import AnnDataManager
from scvi.data.fields import (CategoricalJointObsField, CategoricalObsField,
                              LayerField, NumericalJointObsField)
from scvi.dataloaders import AnnDataLoader
from scvi.model._utils import (_get_batch_code_from_category,
                               _init_library_size, scrna_raw_counts_properties)
from scvi.model.base import BaseModelClass
from scvi.model.base._utils import _de_core
from scvi.utils import setup_anndata_dsp

from multigroup_vi.model.base.training_mixin import MultiGroupTrainingMixin
from multigroup_vi.module.multigroup_vi import MultiGroupVIModule

logger = logging.getLogger(__name__)

Number = Union[int, float]


class MultiGroupVIModel(MultiGroupTrainingMixin, BaseModelClass):
    """
    Implementation of the multiGroupVI model.

    Parameters
    ----------
    adata
        AnnData object that has been registered via :meth:`MultiGroupVIModel.setup_anndata()`.
    n_groups
        Number of groups of samples. Each group corresponds to a different set of
        group-specific latent variables.
    n_hidden
        Number of nodes per hidden layer.
    n_shared_latent
        Dimensionality of the shared latent space.
    n_private_latent
        Dimensionality of the group-specific latent spaces.
    n_layers
        Number of hidden layers used for encoder and decoder NNs.
    wasserstein_penalty
        Strength of the wasserstein-distance penalty discouraging shared information from
        being captured by the private latent spaces.
    """

    def __init__(
        self,
        adata: AnnData,
        n_groups: int,
        n_hidden: int = 128,
        n_shared_latent: int = 10,
        n_private_latent: int = 10,
        n_layers: int = 1,
        wasserstein_penalty: float = 1.0,
    ):
        super(MultiGroupVIModel, self).__init__(adata)

        library_log_means, library_log_vars = _init_library_size(
            self.adata_manager, self.summary_stats["n_batch"]
        )

        # self.summary_stats provides information about anndata dimensions and other tensor info

        self.module = MultiGroupVIModule(
            n_input=self.summary_stats["n_vars"],
            n_hidden=n_hidden,
            n_shared_latent=n_shared_latent,
            n_private_latent=n_private_latent,
            n_layers=n_layers,
            library_log_means=library_log_means,
            library_log_vars=library_log_vars,
            wasserstein_penalty=wasserstein_penalty,
            n_groups=n_groups,
        )
        self._model_summary_string = "Overwrite this attribute to get an informative representation for your model"
        # necessary line to get params that will be used for saving/loading
        self.init_params_ = self._get_init_params(locals())

        logger.info("The model has been initialized")

    @classmethod
    @setup_anndata_dsp.dedent
    def setup_anndata(
        cls,
        adata: AnnData,
        group_key: str,
        batch_key: Optional[str] = None,
        labels_key: Optional[str] = None,
        layer: Optional[str] = None,
        categorical_covariate_keys: Optional[List[str]] = None,
        continuous_covariate_keys: Optional[List[str]] = None,
        **kwargs,
    ) -> Optional[AnnData]:
        """
        %(summary)s.
        Parameters
        ----------
        %(param_adata)s
        %(param_batch_key)s
        %(param_labels_key)s
        %(param_layer)s
        %(param_cat_cov_keys)s
        %(param_cont_cov_keys)s

        Returns
        -------
        %(returns)s
        """
        setup_method_args = cls._get_setup_method_args(**locals())
        anndata_fields = [
            LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=True),
            CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key),
            CategoricalObsField(REGISTRY_KEYS.LABELS_KEY, labels_key),
            CategoricalJointObsField(
                REGISTRY_KEYS.CAT_COVS_KEY, categorical_covariate_keys
            ),
            CategoricalObsField("group", group_key),
            NumericalJointObsField(
                REGISTRY_KEYS.CONT_COVS_KEY, continuous_covariate_keys
            ),
        ]
        adata_manager = AnnDataManager(
            fields=anndata_fields, setup_method_args=setup_method_args
        )
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)

    def differential_expression(
        self,
        adata: Optional[AnnData] = None,
        groupby: Optional[str] = None,
        group1: Optional[Iterable[str]] = None,
        group2: Optional[str] = None,
        idx1: Optional[Union[Sequence[int], Sequence[bool], str]] = None,
        idx2: Optional[Union[Sequence[int], Sequence[bool], str]] = None,
        mode: Literal["vanilla", "change"] = "change",
        delta: float = 0.25,
        batch_size: Optional[int] = None,
        all_stats: bool = True,
        batch_correction: bool = False,
        batchid1: Optional[Iterable[str]] = None,
        batchid2: Optional[Iterable[str]] = None,
        fdr_target: float = 0.05,
        silent: bool = False,
        **kwargs,
    ) -> pd.DataFrame:
        r"""
        A unified method for differential expression analysis.
        Implements `"vanilla"` DE [Lopez18]_ and `"change"` mode DE [Boyeau19]_.
        Parameters
        ----------
        {doc_differential_expression}
        **kwargs
            Keyword args for :meth:`scvi.model.base.DifferentialComputation.get_bayes_factors`
        Returns
        -------
        Differential expression DataFrame.
        """
        adata = self._validate_anndata(adata)

        col_names = adata.var_names
        model_fn = partial(
            self.get_normalized_expression,
            return_numpy=True,
            n_samples=1,
            batch_size=batch_size,
            indices_to_return_shared=[],
        )
        result = _de_core(
            self.get_anndata_manager(adata, required=True),
            model_fn,
            groupby,
            group1,
            group2,
            idx1,
            idx2,
            all_stats,
            scrna_raw_counts_properties,
            col_names,
            mode,
            batchid1,
            batchid2,
            delta,
            batch_correction,
            fdr_target,
            silent,
            **kwargs,
        )

        return result

    @torch.no_grad()
    def get_latent_representation(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        give_mean: bool = True,
        batch_size: Optional[int] = None,
        representation_kind: str = "shared",
        group_number: int = None,
    ) -> np.ndarray:
        """
        Return the shared or private latent representation for each cell.
        Args:
        ----
        adata: AnnData object with equivalent structure to initial AnnData. If `None`,
            defaults to the AnnData object used to initialize the model.
        indices: Indices of cells in adata to use. If `None`, all cells are used.
        give_mean: Give mean of distribution or sample from it.
        batch_size: Mini-batch size for data loading into model. Defaults to
            `scvi.settings.batch_size`.
        representation_kind: Either "shared" or "private" to specify which set of latent
            variables to return.
        group_number: If `representation_kind == "private"`, specifies which set of
            private variables to return (e.g. private variables for group 1, private
            variables for group 2, etc.).
        Returns
        -------
            A numpy array with shape `(n_cells, n_latent)`.
        """
        available_representation_kinds = ["shared", "private"]
        assert representation_kind in available_representation_kinds, (
            f"representation_kind = {representation_kind} is not one of"
            f" {available_representation_kinds}"
        )

        adata = self._validate_anndata(adata)
        data_loader = self._make_data_loader(
            adata=adata,
            indices=indices,
            batch_size=batch_size,
            shuffle=False,
            data_loader_class=AnnDataLoader,
        )
        latent = []
        for tensors in data_loader:
            x = tensors[REGISTRY_KEYS.X_KEY]
            group_labels = tensors["group"]
            outputs = self.module.inference(x=x, group_labels=group_labels)

            if representation_kind == "shared":
                latent_m = outputs["qz_m"]
                latent_sample = outputs["z"]
            elif representation_kind == "private":
                latent_m = outputs["qt_m_all"][
                    :,
                    self.module.n_private_latent
                    * group_number : self.module.n_private_latent
                    * (group_number + 1),
                ]
                latent_sample = outputs["t_all"][
                    :,
                    self.module.n_private_latent
                    * group_number : self.module.n_private_latent
                    * (group_number + 1),
                ]

            if give_mean:
                latent_sample = latent_m

            latent += [latent_sample.detach().cpu()]
        return torch.cat(latent).numpy()

    @torch.no_grad()
    def get_normalized_expression(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        transform_batch: Optional[Sequence[Union[Number, str]]] = None,
        gene_list: Optional[Sequence[str]] = None,
        library_size: Union[float, str] = 1.0,
        n_samples: int = 1,
        n_samples_overall: Optional[int] = None,
        batch_size: Optional[int] = None,
        return_mean: bool = True,
        return_numpy: Optional[bool] = None,
        indices_to_return_shared: list = [],
    ) -> Dict[str, Union[np.ndarray, pd.DataFrame]]:
        """
        Return the normalized (decoded) gene expression.
        Args:
        ----
        adata: AnnData object with equivalent structure to initial AnnData. If `None`,
            defaults to the AnnData object used to initialize the model.
        indices: Indices of cells in adata to use. If `None`, all cells are used.
        transform_batch: Batch to condition on. If transform_batch is:
            - None, then real observed batch is used.
            - int, then batch transform_batch is used.
        gene_list: Return frequencies of expression for a subset of genes. This can
            save memory when working with large datasets and few genes are of interest.
        library_size:  Scale the expression frequencies to a common library size. This
            allows gene expression levels to be interpreted on a common scale of
            relevant magnitude. If set to `"latent"`, use the latent library size.
        n_samples: Number of posterior samples to use for estimation.
        n_samples_overall: The number of random samples in `adata` to use.
        batch_size: Mini-batch size for data loading into model. Defaults to
            `scvi.settings.batch_size`.
        return_mean: Whether to return the mean of the samples.
        return_numpy: Return a `numpy.ndarray` instead of a `pandas.DataFrame`.
            DataFrame includes gene names as columns. If either `n_samples=1` or
            `return_mean=True`, defaults to `False`. Otherwise, it defaults to `True`.
        indices_to_return_shared: If not empty, specifies a set of cells that should be
            decoded using _only_ the models' shared latent variables.
        Returns
        -------
            A dictionary with keys "background" and "salient", with value as follows.
            If `n_samples` > 1 and `return_mean` is `False`, then the shape is
            `(samples, cells, genes)`. Otherwise, shape is `(cells, genes)`. In this
            case, return type is `pandas.DataFrame` unless `return_numpy` is `True`.
        """
        adata = self._validate_anndata(adata)
        if indices is None:
            indices = np.arange(adata.n_obs)
        if n_samples_overall is not None:
            indices = np.random.choice(indices, n_samples_overall)
        data_loader = self._make_data_loader(
            adata=adata,
            indices=indices,
            batch_size=batch_size,
            shuffle=False,
            data_loader_class=AnnDataLoader,
        )

        transform_batch = _get_batch_code_from_category(
            self.get_anndata_manager(adata, required=True), transform_batch
        )

        if gene_list is None:
            gene_mask = slice(None)
        else:
            all_genes = adata.var_names
            gene_mask = [True if gene in gene_list else False for gene in all_genes]

        if n_samples > 1 and return_mean is False:
            if return_numpy is False:
                warnings.warn(
                    "return_numpy must be True if n_samples > 1 and"
                    " return_mean is False, returning np.ndarray"
                )
            return_numpy = True
        if library_size == "latent":
            generative_output_key = "px_rate"
            scaling = 1
        else:
            generative_output_key = "px_scale"
            scaling = library_size

        exprs = []
        for tensors in data_loader:
            x = tensors[REGISTRY_KEYS.X_KEY]
            group_labels = tensors["group"]
            batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]
            per_batch_exprs = []
            for batch in transform_batch:
                if batch is not None:
                    batch_index = torch.ones_like(batch_index) * batch
                inference_outputs = self.module.inference(
                    x=x, group_labels=group_labels
                )
                z = inference_outputs["z"]
                t_all = inference_outputs["t_all"]
                library = inference_outputs["library"]

                generative_outputs = self.module.generative(
                    z=z,
                    t_all=torch.zeros_like(t_all)
                    if set(indices).issubset(set(indices_to_return_shared))
                    else t_all,
                    library=library,
                )

                output = generative_outputs[generative_output_key]
                output = output[..., gene_mask]
                output *= scaling
                output = output.cpu().numpy()
                per_batch_exprs.append(output)

            per_batch_exprs = np.stack(
                per_batch_exprs
            )  # Shape is (len(transform_batch) x batch_size x n_var).
            per_batch_exprs = np.stack(per_batch_exprs)
            exprs += [per_batch_exprs.mean(0)]

        if n_samples > 1:
            # The -2 axis correspond to cells.
            exprs = np.concatenate(exprs, axis=-2)
        else:
            exprs = np.concatenate(exprs, axis=0)
        if n_samples > 1 and return_mean:
            exprs = exprs.mean(0)

        if return_numpy is None or return_numpy is False:
            genes = adata.var_names[gene_mask]
            samples = adata.obs_names[indices]
            exprs = pd.DataFrame(exprs, columns=genes, index=samples)
        return exprs

        
r"""
    Author: Tianjun Wei (tjwei2-c@my.cityu.edu.hk)
    Name: model.py
    Created Date: 2022/10/05
    Modified date: 2023/02/05
    Description: Fine-tuning Partition-aware Item Similarities for Efficient and Scalable Recommendation (FPSR) - CuPy Version
"""
import torch
import cupy as cp
import numpy as np
import scipy.sparse as sp

from cupyx.scipy.sparse.linalg import svds, lobpcg

from recbole.utils import InputType
from recbole.utils.enum_type import ModelType
from recbole.model.abstract_recommender import GeneralRecommender

class FPSR(GeneralRecommender):
    r"""
    Fine-tuning Partition-aware Item Similarities for Efficient and Scalable Recommendation
    """
    input_type = InputType.POINTWISE
    type = ModelType.TRADITIONAL

    def __init__(self, config, dataset):
        r"""
        Model initialization and training.
        """
        super().__init__(config, dataset)
        cp.random.seed(config['seed'])
        # Parameters for W
        self.eigen_dim = config['eigenvectors'] # Num of eigenvectors extracted for W
        self.lambda_ = config['lambda']         # Lambda
        self.solver = config['solver']          # Solver of eigendecomposition

        # Parameters for optimization
        self.rho = config['rho']                # Rho
        self.theta_1 = config['theta_1']        # Theta_1
        self.theta_2 = config['theta_2']        # Theta_2
        self.eta = config['eta']                # Eta
        self.opti_iter = config['opti_iter']    # Number of iteration
        self.tol = config['tol']                # Threshold to filter out small values

        # Parameters for recusrive graph partitioning
        self.tau = config['tau']  # Size ratio of partitions
        
        # Dummy Params
        self.dummy_param = torch.nn.Parameter(torch.zeros(1))   # Dummy pytorch parameters required by Recbole
        self.inter_mat = cp.sparse.csc_matrix(dataset.inter_matrix(form='csr'), dtype=cp.float32)   # User-item interaction matrix
        self.S = sp.dok_matrix((self.n_items, self.n_items), dtype=np.float32)  # Paramater S

        # TRAINING PROCESS
        # Calaulate W and generate first split
        first_split = self.update_W()
        # Recursive paritioning and item similarity modeling in partition 1
        self.update_S(np.arange(self.n_items)[np.where(first_split)]) 
        # Recursive paritioning and item similarity modeling in partition 2
        self.update_S(np.arange(self.n_items)[np.where(~first_split)])
        
        self.S = cp.sparse.csr_matrix(self.S.tocsr(), dtype=cp.float32)

    def _degree(self, inter_mat=None, axis=0, exp=-0.5):
        r"""
        Degree of nodes
        """
        if inter_mat is None:
            inter_mat = self.inter_mat
        axis_sum = inter_mat.sum(axis=axis)
        d_inv = cp.power(axis_sum, exp).flatten()
        d_inv[cp.isinf(d_inv)] = 0.
        return d_inv

    def _svd(self, mat, k):
        r"""
        Truncated singular value decomposition (SVD)
        """
        if self.solver == 'lobpcg':
            _, V = lobpcg(mat.T @ mat, cp.random.rand(mat.shape[1], k), largest=True)
        else:
            _, _, V = svds(mat, 4*k, maxiter=10000)
            V = V.T
        return V[:, :k]

    def _norm_adj(self, item_list=None):
        r"""
        Normalized adjacency matrix
        """
        if item_list is None:
            return cp.sparse.diags(self._degree(axis=1)) @ self.inter_mat @ cp.sparse.diags(self.d_i.flatten())
        else:
            inter_mat = self.inter_mat[:, item_list]
            return cp.sparse.diags(self._degree(inter_mat, axis=1)) @ inter_mat @ cp.sparse.diags(self.d_i.flatten()[item_list])
            
    def update_W(self):
        r"""
        Update W
        (Only store V and D_I instead of W)
        """
        self.d_i = self._degree(axis=0).reshape(-1, 1)
        self.d_i_inv = self._degree(axis=0, exp=0.5).reshape(1, -1)
        self.V = self._svd(self._norm_adj(), self.eigen_dim)
        return cp.asnumpy(self.V[:, 1] >= 0)

    def partitioning(self, item_list):
        r"""
        Graph biparitioning
        """
        V = self._svd(self._norm_adj(item_list), 2)
        split = cp.asnumpy(V[:, 1] >= 0)
        if split.sum() == split.shape[0] or split.sum() == 0:
            split = cp.asnumpy(V[:, 1] >= cp.median(V[:, 1]))
        return split

    def update_S(self, item_list):
        r"""
        Update S (recursive)
        """
        if item_list.shape[0] <= self.tau * self.n_items:
            # If the partition size is samller than size limit, model item similarity for this partition.
            comm_inter = self.inter_mat[:, item_list]
            comm_inter = comm_inter.T @ comm_inter
            comm_ae = self.item_similarity(
                comm_inter,
                self.V[item_list, :],
                self.d_i[item_list, :],
                self.d_i_inv[:, item_list]
            )
            comm_ae.data *= comm_ae.data >= self.tol    # Filter out small values
            comm_ae.eliminate_zeros()
            self.S._update(
                dict(
                    zip(
                        zip(item_list[comm_ae.row], item_list[comm_ae.col]),
                        comm_ae.data
                    )
                )
            )
            print("Node Num: {:5d} Weight Matrix Non-zero: {:8d}".format(comm_ae.shape[0], comm_ae.nnz))
        else:
            # If the partition size is larger than size limit, perform graph partitioning on this partition.
            split = self.partitioning(item_list)
            self.update_S(item_list[np.where(split)])
            self.update_S(item_list[np.where(~split)])
    
    def item_similarity(self, inter_mat, V, d_i, d_i_inv) -> sp.coo_matrix:
        r"""
        Similarity modeling in each partition
        """
        # Initialize
        Q_hat = (inter_mat.todense() + self.theta_2 * cp.diag(cp.power(d_i_inv.flatten(), 2)) + self.eta).astype(cp.float32)
        Q_inv = cp.linalg.inv(Q_hat + (self.rho) * cp.identity(inter_mat.shape[0])).astype(cp.float32)
        Z_aux = (Q_inv @ Q_hat @ (cp.identity(inter_mat.shape[0]) - self.lambda_ * d_i * V @ V.T * d_i_inv)).astype(cp.float32)
        Phi = cp.zeros_like(Q_inv, dtype=cp.float32)
        S = cp.zeros_like(Q_inv, dtype=cp.float32)
        del Q_hat
        for _ in range(self.opti_iter):
            # Iteration
            Z_tilde = Z_aux + Q_inv @ (self.rho *(S - Phi))
            gamma = cp.diag(Z_tilde) / (cp.diag(Q_inv) + 1e-10)
            Z = Z_tilde - Q_inv * gamma                             # Update Z
            S = (Z + Phi - self.theta_1 / self.rho).clip(0)         # Update S 
            Phi += Z - S                                            # Update Phi
        return sp.coo_matrix(cp.asnumpy(S))

    def forward(self):
        r"""
        Abstract method of GeneralRecommender in RecBole (not used)
        """
        pass

    def calculate_loss(self, interaction):
        r"""
        Abstract method of GeneralRecommender in RecBole (not used)
        """
        return torch.nn.Parameter(torch.zeros(1))

    def predict(self, interaction):
        r"""
        Abstract method of GeneralRecommender in RecBole (not used)
        """
        raise NotImplementedError

    def full_sort_predict(self, interaction):
        r"""
        Recommend items for the input users
        """
        user = cp.array(interaction[self.USER_ID].cpu().numpy())
        user_interactions = self.inter_mat[user, :].toarray()
        r = user_interactions @ self.S
        r += self.lambda_ * user_interactions * self.d_i.T @ self.V @ self.V.T * self.d_i_inv
        return torch.from_numpy(cp.asnumpy(r.flatten()))

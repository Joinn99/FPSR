r"""
    Author: Tianjun Wei (tjwei2-c@my.cityu.edu.hk)
    Name: model.py
    Created Date: 2022/10/05
    Modified date: 2023/02/01
    Description: Fine-tuning Partition-aware Item Similarities for Efficient and Scalable Recommendation (FPSR) - PyTorch Version
"""
import torch
import numpy as np

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
        self.inter = dataset.inter_matrix(form='coo')   # User-item interaction matrix
        self.inter = torch.sparse_coo_tensor(
                            torch.LongTensor(np.array([self.inter.row, self.inter.col])),
                            torch.FloatTensor(self.inter.data),
                            size=self.inter.shape, dtype=torch.float
                        ).coalesce().to(self.device)
        
        self.S_indices = []
        self.S_values = []

        # TRAINING PROCESS
        # Calaulate W and generate first split
        first_split = self.update_W()
        # Recursive paritioning and item similarity modeling in partition 1
        self.update_S(torch.arange(self.n_items, device=self.device)[torch.where(first_split)[0]])
        # Recursive paritioning and item similarity modeling in partition 2
        self.update_S(torch.arange(self.n_items, device=self.device)[torch.where(~first_split)[0]])
        
        self.S = torch.sparse_coo_tensor(indices=torch.cat(self.S_indices, dim=1),
                                         values=torch.cat(self.S_values, dim=0),
                                         size=(self.n_items, self.n_items)).coalesce().T

    def _degree(self, inter_mat=None, dim=0, exp=-0.5):
        r"""
        Degree of nodes
        """
        if inter_mat is None:
            inter_mat = self.inter
        d_inv = torch.nan_to_num(torch.sparse.sum(inter_mat,dim=dim).to_dense().pow(exp), nan=0, posinf=0, neginf=0)
        return d_inv

    def _svd(self, mat, k):
        r"""
        Truncated singular value decomposition (SVD)
        """
        if self.solver == 'lobpcg':
            _, V = torch.lobpcg(A=torch.sparse.mm(mat.T, mat), X=torch.rand(mat.shape[1], k).to(self.device))
        else:
            _, _, V = torch.svd_lowrank(mat, q=max(4*k, 32), niter=10)
        return V[:, :k]

    def _norm_adj(self, item_list=None):
        r"""
        Normalized adjacency matrix
        """
        if item_list is None:
            vals = self.inter.values() * self.d_i[self.inter.indices()[1]].squeeze()
            return torch.sparse_coo_tensor(
                            self.inter.indices(),
                            self._degree(dim=1)[self.inter.indices()[0]] * vals,
                            size=self.inter.shape, dtype=torch.float
                        ).coalesce()
        else:
            inter = self.inter.index_select(dim=1, index=item_list).coalesce()
            vals = inter.values() * self.d_i[item_list][inter.indices()[1]].squeeze()
            return torch.sparse_coo_tensor(
                            inter.indices(),
                            self._degree(inter, dim=1)[inter.indices()[0]] * vals,
                            size=inter.shape, dtype=torch.float
            ).coalesce()
    

    def update_W(self) -> torch.Tensor:
        r"""
        Update W
        (Only store V and D_I instead of W)
        """
        self.d_i = self._degree(dim=0).reshape(-1, 1)
        self.d_i_inv = self._degree(dim=0, exp=0.5).reshape(1, -1)
        self.V = self._svd(self._norm_adj(), self.eigen_dim)
        return self.V[:, 1] >= 0

    def partitioning(self, item_list) -> torch.Tensor:
        r"""
        Graph biparitioning
        """
        V = self._svd(self._norm_adj(item_list), 2)
        split = V[:, 1] >= 0
        if split.sum() == split.shape[0] or split.sum() == 0:
            split = V[:, 1] >= torch.median(V[:, 1])
        return split

    def update_S(self, item_list) -> None:
        r"""
        Update S (recursive)
        """
        if item_list.shape[0] <= self.tau * self.n_items:
            # If the partition size is samller than size limit, model item similarity for this partition.
            comm_inter = self.inter.index_select(dim=1, index=item_list).to_dense()
            comm_inter = torch.mm(comm_inter.T, comm_inter)
            comm_ae = self.item_similarity(
                comm_inter,
                self.V[item_list, :],
                self.d_i[item_list, :],
                self.d_i_inv[:, item_list]
            )
            comm_ae = torch.where(comm_ae >= self.tol, comm_ae, 0).to_sparse_coo()
            self.S_indices.append(item_list[comm_ae.indices()])
            self.S_values.append(comm_ae.values())
            print("Node Num: {:5d} Weight Matrix Non-zero: {:8d}".format(comm_ae.shape[0], comm_ae._nnz()))
        else:
            # If the partition size is larger than size limit, perform graph partitioning on this partition.
            split = self.partitioning(item_list)
            self.update_S(item_list[torch.where(split)[0]])
            self.update_S(item_list[torch.where(~split)[0]])
    
    def item_similarity(self, inter_mat, V, d_i, d_i_inv) -> torch.Tensor:
        r"""
        Similarity modeling in each partition
        """
        # Initialize
        Q_hat = inter_mat + self.theta_2 * torch.diag(torch.pow(d_i_inv.squeeze(), 2)) + self.eta
        Q_inv = torch.inverse(Q_hat + self.rho * torch.eye(inter_mat.shape[0], device=self.device))
        Z_aux = (Q_inv @ Q_hat @ (torch.eye(inter_mat.shape[0], device=self.device) - self.lambda_ * d_i * V @ V.T * d_i_inv))
        del Q_hat
        Phi = torch.zeros_like(Q_inv, device=self.device)
        S = torch.zeros_like(Q_inv, device=self.device)
        for _ in range(self.opti_iter):
            # Iteration
            Z_tilde = Z_aux + Q_inv @ (self.rho * (S - Phi))
            gamma = torch.diag(Z_tilde) / (torch.diag(Q_inv) + 1e-10)
            Z = Z_tilde - Q_inv * gamma                                 # Update Z
            S = torch.clip(Z + Phi - self.theta_1 / self.rho, min=0)    # Update S 
            Phi += Z - S                                                # Update Phi
        return S

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

    def full_sort_predict(self, interaction) -> torch.Tensor:
        r"""
        Recommend items for the input users
        """
        user = self.inter.index_select(dim=0, index=interaction[self.USER_ID]).to_dense()
        r = torch.sparse.mm(self.S, user.T).T
        r += self.lambda_ * user * self.d_i.T @ self.V @ self.V.T * self.d_i_inv
        return r
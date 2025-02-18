""" Low rank laplace approximation tools
"""
from typing import Literal, Optional, Union, Iterable, Tuple, Callable

import torch
import torch.nn as nn
from laplace import KronLaplace, FullLaplace, DiagLaplace

from projector.projector1d import get_jacobian
from utils import iterator_wise_quadratic_form, iterator_wise_matmul, \
    flatten_batch_and_target_dimension


class InvPsi():
    """Classes that inherit from this class should implement `Psi_times_W` 
    `functional_covariance` and `quadratic_form`.
    """
    def Psi_times_W(self, W: torch.Tensor) -> torch.Tensor:
        """ Returns Psi x W """
        raise NotImplementedError

    def Sigma(self, J_X: torch.Tensor) -> torch.Tensor:
        """Returns J_X @ Psi @ J_X.T"""
        raise NotImplementedError

    def Sigma_batchwise(self, J_X: torch.Tensor) -> torch.Tensor:
        # need shape of type batchxtargetxparmeters
        assert len(J_X.shape) == 3
        J_X_T = J_X.transpose(dim0=0, dim1=2)
        Psi_times_J_X_T = self.Psi_times_W(W=J_X_T).view(J_X_T.shape)
        return torch.einsum('btp,pTb -> btT',J_X, Psi_times_J_X_T)



    def Sigma_iterative(self, create_J_X_iterator: Callable[[],Iterable]) -> torch.Tensor:
        return iterator_wise_quadratic_form(quadratic_form=self.bilinear_Psi_form,
                                            create_iterator=create_J_X_iterator)

    def quadratic_form(self, W: torch.Tensor) -> torch.Tensor:
        """ Returns W^T @ inv_Psi @ W"""
        raise NotImplementedError

    def bilinear_Psi_form(self, W1: torch.Tensor, W2: torch.Tensor):
        """Computes W1 @ Psi @ W2.T"""
        # W1 and W2 may only differ in the batch dimension
        # and should be 2D or 3D
        if W1.ndim ==3:
            assert W2.size(1) == W1.size(1)
            assert W2.size(-1) == W1.size(-1)
        else:
            assert W1.ndim == 2 and W2.ndim == 2
            assert W2.size(-1) == W1.size(-1)
        flat_W1 = flatten_batch_and_target_dimension(W1)
        flat_W2 = flatten_batch_and_target_dimension(W2)
        Psi_times_flat_W2 = self.Psi_times_W(W=flat_W2.T)
        return flat_W1 @ Psi_times_flat_W2

    def Sigma_svd(self, J_X: Union[torch.Tensor, Callable[[],Iterable]],
                  singular_vectors: Literal['left', 'right']='left'
                  ) -> Tuple[torch.Tensor, torch.Tensor]:
        if type(J_X) is torch.Tensor:
            Sigma = self.Sigma(J_X)
        else:
            Sigma = self.Sigma_iterative(create_J_X_iterator=J_X)
        left_U, Lamb, right_U_h =\
            torch.linalg.svd(Sigma, full_matrices=True)
        if singular_vectors == 'left':
            U = left_U
        else:
            U = right_U_h.T
        return U, Lamb

        



class FullInvPsi(InvPsi):
    def __init__(self, inv_Psi: Union[torch.Tensor, FullLaplace]) -> None:
        if type(inv_Psi) is torch.Tensor:
            self.inv_Psi_matrix = inv_Psi
        else:
            assert type(inv_Psi) is FullLaplace
            self.inv_Psi_matrix = inv_Psi.posterior_precision
        

    def Psi_times_W(self, W: torch.Tensor) -> torch.Tensor:
        assert self.inv_Psi_matrix.size(0) == W.size(0), "Size mismatch"
        # flatten W to make it 2D
        flat_W = W.reshape(W.size(0), -1)
        # multiply with inv(inv_Psi) by solving linear system
        Psi_times_flat_W = torch.linalg.solve(self.inv_Psi_matrix, flat_W) 
        # restore shape
        return Psi_times_flat_W.view(W.shape)

        
    def Sigma(self, J_X: torch.Tensor) -> torch.Tensor:
        J_X = flatten_batch_and_target_dimension(J_X)
        Psi_times_J_X_T = self.Psi_times_W(J_X.T)
        return J_X @ Psi_times_J_X_T



    def quadratic_form(self, W: torch.Tensor) -> torch.Tensor: 
        if W.ndim > 2:
            # flatten into 2D Tensor
            W = W.view(-1, W.size(-1))
        return W.T @ self.inv_Psi_matrix @ W

class HalfInvPsi(FullInvPsi):
    """Defines inv_Psi via V^T@V + prior_precision * I """
    def __init__(self, V: Union[torch.Tensor, Callable[[],Iterable]], prior_precision: float) -> None:
        if type(V) is torch.Tensor:
            self.V_from_iterator = False
            self.V = flatten_batch_and_target_dimension(V)
        else:
            assert callable(V)
            self.V_from_iterator = True
            self.V = self.flattened_V_iterator(V)
        self.prior_precision = prior_precision

    @staticmethod
    def flattened_V_iterator(V_iterator):
        def flat_it():
            for v in V_iterator():
                yield flatten_batch_and_target_dimension(v)
        return flat_it
        

    @property
    def full_V(self) -> torch.Tensor:
        if not self.V_from_iterator:
            return self.V
        else:
            v_stack = []
            for v in self.V():
                v_stack.append(v)
            return torch.concat(v_stack, dim=0)


    @property
    def inv_Psi_matrix(self):
        V = self.full_V
        H = V.T @ V
        return H + self.prior_precision * torch.eye(H.size(0)).to(V.device)

     

    def quadratic_form(self, W: torch.Tensor) -> torch.Tensor: 
        if W.ndim > 2:
            if not W.ndim == 3:
                # flatten into 2D Tensor
                W = W.view(-1, W.size(-1))
        if self.V_from_iterator:
            sum_matrix_products = 0 
            for v in self.V():
                v_times_W = v @ W
                sum_matrix_products += v_times_W.T @ v_times_W
            assert type(sum_matrix_products) is torch.Tensor,\
                  "no summation happened"
            return sum_matrix_products + self.prior_precision * W.T @ W
        else:
            V_times_W = self.V @ W
            return V_times_W.T @ V_times_W + self.prior_precision * W.T @ W


class DiagInvPsi(FullInvPsi):
    """Wraps instances of the class DiagLaplace into the InvPsi class.
    """
    def __init__(self, inv_Psi: DiagLaplace):
        self.inv_Psi = inv_Psi

    @property
    def posterior_variance(self) -> torch.Tensor:
        return self.inv_Psi.posterior_variance

    @property
    def posterior_precision(self) -> torch.Tensor:
        return self.inv_Psi.posterior_precision

    @property
    def inv_Psi_matrix(self):
        return torch.diag(self.posterior_precision)

    def Psi_times_W(self, W: torch.Tensor) -> torch.Tensor:
        assert self.posterior_variance.size(0) == W.size(0), "Size mismatch"
        # flatten W to make it 2D
        flat_W = W.reshape(W.size(0), -1)
        Psi_times_flat_W = self.posterior_variance[:,None] * flat_W
        # restore shape
        return Psi_times_flat_W.view(W.shape)

    def _inv_Psi_times_W(self, W: torch.Tensor) -> torch.Tensor:
        assert self.posterior_precision.size(0) == W.size(0), "Size mismatch"
        # flatten W to make it 2D
        flat_W = W.reshape(W.size(0), -1)
        inv_Psi_times_flat_W = self.posterior_precision[:,None] * flat_W
        # restore shape
        return inv_Psi_times_flat_W.view(W.shape)


    def quadratic_form(self, W):
        if W.ndim > 2:
            # flatten into 2D Tensor
            W = W.view(-1, W.size(-1))
        return W.T @ self._inv_Psi_times_W(W=W)

class KronInvPsi(InvPsi):
    """Wraps instances of the class KronLaplace into the InvPsi class.
    """
    def __init__(self, inv_Psi: KronLaplace) -> None:
        self.inv_Psi = inv_Psi

    def Psi_times_W(self, W: torch.Tensor) -> torch.Tensor:
        # flatten W to make it 2D
        flat_W = W.reshape(W.size(0), -1)
        # multiply with flat_W by using bmm
        # transpose to match dimensions
        Psi_times_flat_W = self.inv_Psi.posterior_precision.bmm(flat_W.T, exponent=-1).T
        # restore shape
        return Psi_times_flat_W.view(W.shape)
        
        
    def Sigma(self, J_X: torch.Tensor) -> torch.Tensor:
        if J_X.ndim == 2:
            # add a class dimension
            J_X = J_X.view(-1, self.inv_Psi.n_outputs, self.inv_Psi.n_params)
        else:
            assert J_X.ndim == 3
        return self.inv_Psi.functional_covariance(Js=J_X)

    def quadratic_form(self, W: torch.Tensor) -> torch.Tensor:
        # flatten W to make it 2D
        flat_W = W.view(W.size(0), -1)
        # multiply with flat_W by using bmm
        # transpose to match dimensions
        flat_W_T_times_Psi_inv = self.inv_Psi.posterior_precision.bmm(flat_W.T,
                                                                exponent=1.0)
        if flat_W.size(-1) == 1:
            # reinsert dimension that is deleted in this case
            flat_W_T_times_Psi_inv = flat_W_T_times_Psi_inv.unsqueeze(dim=0)
        return flat_W_T_times_Psi_inv @ W
                                                



def compute_Sigma(IPsi: InvPsi,
                  J_X: Union[torch.Tensor, Callable[[], Iterable]]) -> torch.Tensor:
    if type(J_X) is torch.Tensor:
        return IPsi.Sigma(J_X=J_X)
    else:
        return IPsi.Sigma_iterative(create_J_X_iterator=J_X)

def compute_Sigma_s(U: torch.Tensor, Lamb: torch.Tensor,
                    s: float) -> torch.Tensor:
    U_s = U[:,:s]
    Lamb_s = Lamb[:s]
    return U_s @ torch.diag(Lamb_s) @ U_s.T

def compute_optimal_P(IPsi: InvPsi, J_X: Union[torch.Tensor, Callable[[], Iterable]],
                      U: torch.Tensor, s: Optional[int]=None,
                      Q: Optional[torch.Tensor]=None) -> torch.Tensor:
        if s is not None:
            U_s = U[:,:s]
        else:
            U_s = U
        if type(J_X) is torch.Tensor:
            W = J_X.T @ U_s
        else:
            W = iterator_wise_matmul(J_X, U_s, transpose_a=True, iteration_dim=1)
        P = IPsi.Psi_times_W(W=W)
        if Q:
            P = P @ Q
        return P
    

def compute_Sigma_P(P: torch.Tensor, IPsi: InvPsi,
                    J_X: Union[torch.Tensor, Callable[[], Iterable]],
                    s_iterable: Optional[Iterable[int]] = None,
                    # eps: float = 1e-10,
                    ) -> Union[torch.Tensor, Callable[[], Iterable]]:
    inv = torch.linalg.inv
    # if eps > 0:
    #     s_max = s_iterable[-1]
    #     P = torch.concat((P, torch.zeros((P.size(0), s_max - P.size(1))).to(P.device)), dim=-1) \
    #         + eps * torch.randn(P.size(0), s_max).to(P.device)
    P_T_inv_Psi_P = IPsi.quadratic_form(W=P)
    if type(J_X) is torch.Tensor:
        if len(J_X.shape) > 2:
            J_X = flatten_batch_and_target_dimension(J_X)
        J_X_times_P = J_X @ P
    else:
        J_X_times_P = iterator_wise_matmul(J_X, P, transpose_a=False,
                                           iteration_dim=0)
    if s_iterable is None:
        return J_X_times_P @ inv(P_T_inv_Psi_P) @ J_X_times_P.T
    else:
        def create_Sigma_P_iterator():
            for s in s_iterable:
                J_X_times_P_s = J_X_times_P[:,:s]
                P_s_T_inv_Psi_P_s_T = P_T_inv_Psi_P[:s,:s]
                yield J_X_times_P_s @ inv(P_s_T_inv_Psi_P_s_T) @ J_X_times_P_s.T
        return create_Sigma_P_iterator





class IPsi_predictive():
    """Computes predictions and covariances for the
    linearized model produced from `model`, `IPsi` and `P` (optional). If `P`
    is not `None` the projected model is used, otherwise the full model is used.
    A `chunk_size`
    can be specified for the internal computation of the jacobian.
    If `regression_likelihood_sigma` is specified the (homoscedastic) variance
    of the likelihood is accounted for. This should only be done for regression
    problems.
    """
    def __init__(
        self,
        model: nn.Module,
        IPsi: InvPsi,
        P: Optional[torch.Tensor],
        chunk_size: Optional[int]  = None,
        regression_likelihood_sigma: Optional[float] = None,
    ):
        # predictions
        self.model = model.eval()
        self.IPsi = IPsi
        self.P = P
        self.chunk_size = chunk_size
        self.regression_likelihood_sigma = regression_likelihood_sigma
        if self.P is not None:
            self._P_T_inv_Psi_P = IPsi.quadratic_form(W=P)

    def __call__(
            self,
            X: torch.Tensor,
            s: Optional[int] = None,
            eps: float = 1e-10,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        predictions = self.model(X).detach()
        # uncertainties
        J_X = get_jacobian(
            self.model,
            X=X,
            is_classification=False,
            chunk_size=self.chunk_size
        ).detach()
        if self.P is not None:
            assert self._P_T_inv_Psi_P is not None
            # batch_size x target dimension x number of parameters
            assert len(J_X.shape) == 3
            J_X_P = torch.einsum('btp,ps->bts', J_X, self.P[:, :s])
            P_T_inv_Psi_P = self._P_T_inv_Psi_P[:s, :s]
            P_T_inv_Psi_P += eps \
                * torch.eye(P_T_inv_Psi_P.size(-1)).to(P_T_inv_Psi_P.device)
            inv_P_T_inv_Psi_P = torch.linalg.inv(P_T_inv_Psi_P)
            variances = torch.einsum(
                'bts,sS,STb->btT',
                J_X_P,
                inv_P_T_inv_Psi_P,
                J_X_P.transpose(0, -1)
            )
        else:
            variances = self.IPsi.Sigma_batchwise(J_X=J_X)
        
        if self.regression_likelihood_sigma is not None:
            assert len(variances.shape) == 3 # batch x target x target
            variances += self.regression_likelihood_sigma**2 \
                * torch.eye(variances.size(-1)).to(variances.device)

        return predictions, variances
    



import torch
import torch.nn as nn


class AnalyticLayer(nn.Module):
    """
    An analytic projection layer with internal clamping
    to prevent numerical instability during purification.
    """
    def __init__(self, num_electrons, restricted_scheme, purification_iterations):
        super().__init__()
        self.num_electrons = float(num_electrons)
        self.restricted_scheme = bool(restricted_scheme)
        self.batch_trace = lambda X: X.diagonal(dim1=-2, dim2=-1).sum(-1)
        self.purification_iterations = int(purification_iterations)

    # ----------------------------- utils -----------------------------

    def _assert_finite(self, X, tag: str):
        if not torch.isfinite(X).all():
            mn = torch.nanmin(X).item()
            mx = torch.nanmax(X).item()
            raise RuntimeError(f"[AnalyticLayer] {tag} has nonfinite values: min={mn}, max={mx}")

    def _solve_tri(self, A, B, upper=False, transpose=False, left=True):
        """
        Solve A X = B where A is triangular.
        Uses torch.linalg.solve_triangular if available, else torch.triangular_solve.
        """
        if hasattr(torch.linalg, "solve_triangular"):
            return torch.linalg.solve_triangular(
                A, B, upper=upper, left=left, unitriangular=False, transpose=transpose
            )
        # Fallback for older PyTorch: triangular_solve always solves left: A X = B
        if not left:
            # Right solve: X A = B  ==>  A^T X^T = B^T
            Xt, info = torch.triangular_solve(
                B.transpose(-1, -2), A.transpose(-1, -2),
                upper=not upper, transpose=not transpose, unitriangular=False
            )
            return Xt.transpose(-1, -2)
        X, info = torch.triangular_solve(
            B, A, upper=upper, transpose=transpose, unitriangular=False
        )
        return X

    # ---------------------- trace rescale (grads ON) ----------------------

    def _force_trace(self, density_matrix: torch.Tensor, S: torch.Tensor, eps: float = 1e-6):
        P = density_matrix
        tr = torch.clamp(self.batch_trace(P @ S), min=eps)
        scale_factor = (self.num_electrons / tr).view(-1, 1, 1)
        P = scale_factor * P
        return P

    # --------- Generalized McWeeny in AO metric (no eigendecomp) ---------

    def _purify2(self, density_matrix: torch.Tensor, S: torch.Tensor):
        P = density_matrix
        if self.restricted_scheme:
            P = 0.5 * P  # work with 1e- density

        for _ in range(self.purification_iterations):
            PSP = P @ S @ P
            SP  = S @ P
            P   = 3.0 * PSP - 2.0 * (PSP @ SP)

        if self.restricted_scheme:
            P = 2.0 * P  # back to spin-summed

        return 0.5 * (P + P.transpose(-1, -2))

    # ------------- Orthonormal-basis McWeeny (via Cholesky) --------------

    def _purify(self, density_matrix: torch.Tensor, S: torch.Tensor,
                eps: float = 1e-10, damp: float = 0.5):
        P = density_matrix

        # Q = L^T P L
        Q, L = self.get_occupation(P, S, eps=eps)

        # Restricted: operate on 1e- density (R = Q/2). Unrestricted: factor = 1.
        restricted_factor = 0.5 if self.restricted_scheme else 1.0
        Q = restricted_factor * Q

        # McWeeny purification loop in orthonormal basis
        Q_M = Q.clone()
        for _ in range(self.purification_iterations):
            Qsq  = Q_M @ Q_M
            Qcub = Qsq @ Q_M
            Q_M  = 3.0 * Qsq - 2.0 * Qcub

        # Damped update and restore spin-summed scale if restricted
        inv_rf = 1.0 / restricted_factor  # 2.0 if restricted, 1.0 if not
        Q = inv_rf * ((1.0 - damp) * Q + damp * Q_M)

        # Re-enforce electron count in Q
        trQ  = torch.einsum('...ii->...', Q)
        beta = (self.num_electrons / torch.clamp(trQ, min=1e-14)).view(-1, 1, 1)
        Q    = 0.5 * (beta * Q + (beta * Q).transpose(-1, -2))

        # Back to AO without explicit inverse: P = L^{-T} Q L^{-1}
        # Solve (L^T) X = Q  -> X = _solve_tri(L, Q, transpose=True)
        X = self._solve_tri(L, Q, upper=False, transpose=True, left=True)
        # Then (L^T) P^T = X^T  -> P^T = _solve_tri(L, X^T, transpose=True)
        Pt = self._solve_tri(L, X.transpose(-1, -2), upper=False, transpose=True, left=True)
        P_pure = Pt.transpose(-1, -2)
        return 0.5 * (P_pure + P_pure.transpose(-1, -2))

    # ---------- Orthonormalization helper: returns Q = L^T P L and L ----------

    def get_occupation(self, P: torch.Tensor, S: torch.Tensor, eps: float = 1e-10):
        P    = 0.5 * (P + P.transpose(-1, -2))
        Ssym = 0.5 * (S + S.transpose(-1, -2))
        I    = torch.eye(S.shape[-1], dtype=S.dtype, device=S.device)
        L    = torch.linalg.cholesky(Ssym + eps * I)  # S = L L^T
        Q    = L.transpose(-1, -2) @ P @ L
        Q    = 0.5 * (Q + Q.transpose(-1, -2))
        return Q, L

    # -------------------------------- forward --------------------------------

    def forward(self, density_matrix: torch.Tensor, S: torch.Tensor = None, purify: bool = False):
        self._assert_finite(density_matrix, "head_output")

        # Symmetrize raw output
        P = 0.5 * (density_matrix + density_matrix.transpose(-1, -2))

        if S is None:
            return P

        # Pre-trace scaling
        P = self._force_trace(P, S)
        self._assert_finite(P, "after_pre_trace_scale")

        if purify:
            P = self._purify2(P, S)   # or: P = self._purify2(P, S)
            self._assert_finite(P, "after_purification")
            # Optional hygiene (usually tiny): P = self._force_trace(P, S)
            
        P = self._force_trace(P, S)
        self._assert_finite(P, "after_second_trace_forcing")    

        return P
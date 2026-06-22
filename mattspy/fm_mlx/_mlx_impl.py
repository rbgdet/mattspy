import mlx.core as mx
import mlx.optimizers as optim
import numpy as np
import mlx.nn as nn
from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.utils import check_random_state
from sklearn.utils.validation import validate_data
from sklearn.utils.multiclass import type_of_target
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import LabelEncoder

from mattspy.json import EstimatorToFromJSONMixin


def _lowrank_twoway_term(x, vmat):
    x_bf = x.astype(mx.bfloat16)
    vmat_bf = vmat.astype(mx.bfloat16)

    fterm = mx.einsum("np,pkc->nkc", x_bf, vmat_bf).astype(mx.float32)
    sterm = mx.einsum("np,pkc->nkc", x_bf**2, vmat_bf**2).astype(mx.float32)
    # P, K, C = vmat_bf.shape

    # vmat_flat = vmat_bf.reshape(P, K * C)
    # vmat_sq_flat = (vmat_bf**2).reshape(P, K * C)

    # fterm = mx.matmul(x_bf, vmat_flat).reshape(-1, K, C).astype(mx.float32)
    # sterm = mx.matmul(x_bf**2, vmat_sq_flat).reshape(-1, K, C).astype(mx.float32)
    return 0.5 * mx.sum(fterm**2 - sterm,
                         axis=1)


def _linear_term(x, w):
    # return mx.matmul(x, w)
    return mx.einsum("np,p...->n...", x, w)


def _fm_eval(x, w0, w, vmat):
    return w0 + _linear_term(x, w) + _lowrank_twoway_term(x, vmat)


@mx.compile
def _extract_fm_params(params, n_features, rank, n_classes):
    if n_classes is None or n_classes < 1:
        w0 = params[0]
        w = params[1 : 1 + n_features]
        vmat = params[1 + n_features :].reshape((n_features, rank))
    else:
        w0 = params[:n_classes]
        w = params[n_classes : n_classes + n_features * n_classes].reshape(
            (n_features, n_classes)
        )
        vmat = params[n_classes + n_features * n_classes :].reshape(
            (n_features, rank, n_classes)
        )

    return w0, w, vmat


@mx.compile
def _combine_fm_params(w0, w, vmat):
    return mx.concatenate([mx.atleast_1d(w0).flatten(), w.flatten(), vmat.flatten()])


@mx.compile
def _mlx_logits(params, X):
    w0, w, vmat = params["w0"], params["w"], params["vmat"]
    logits = _fm_eval(X, w0, w, vmat)
    return logits


_mlx_proba = mx.compile(
    lambda params, X: nn.softmax(_mlx_logits(params, X), axis=-1),
)

_mlx_log_proba = mx.compile(
    lambda params, X: nn.log_softmax(_mlx_logits(params, X), axis=-1),
)

_mlx_predict = mx.compile(
    lambda params, X: mx.argmax(_mlx_logits(params, X), axis=-1),
)


def _mlx_loss_func(params, X, y, lambda_v, lambda_w):
    w0, w, vmat = params["w0"], params["w"], params["vmat"]
    logits = _fm_eval(X, w0, w, vmat)

    loss = mx.mean(nn.losses.cross_entropy(logits, y))

    if lambda_v > 0:
        loss += lambda_v * mx.sum(vmat**2)

    if lambda_w > 0:
        loss += lambda_w * mx.sum(w**2)

    return loss


_grad_mlx_loss_func = mx.compile(
    mx.grad(_mlx_loss_func),
)
_value_and_grad_mlx_loss_func = mx.compile(
    mx.value_and_grad(_mlx_loss_func),
)


def _call_in_batches_maybe(self, func, X):
    if self.batch_size is not None:
        vals = []
        for start in range(0, X.shape[0], self.batch_size):
            end = min(start + self.batch_size, X.shape[0])
            Xb = X[start:end, :]
            vals.append(func(self.params_, Xb))
        return mx.concatenate(vals, axis=0)
    else:
        return func(self.params_, X)


class _LabelEncoder(EstimatorToFromJSONMixin, LabelEncoder):
    json_attributes_ = ("classes_",)


class FMClassifier(EstimatorToFromJSONMixin, ClassifierMixin, BaseEstimator):
    r"""A Factorization Machine classifier.

    The FM model for the logits for class c is

        logit_c = w0_c + w_c^T * X + \sum_i \sum_{j=i+1} v_{c,i}^T v_{c,j} x_i x_j

    Parameters
    ----------
    rank : int, optional
        The dimension of the low-rank approximation to the
        two-way interaction terms.
    random_state : int, numpy RNG instance, or None
        The RNG to use for parameter initialization.
    batch_size : int, optional
        The number of examples to use when fitting the estimator
        and making predictions. The value None indicates to use all
        examples. This parameter is ignored if the solver is set to `lbfgs`.
    lambda_v : float, optional
        The L2 regularization strength to use for the low-rank embedding
        matrix.
    lambda_w : float, optional
        The L2 regularization strength to use for the linear terms.
    init_scale : float, optional
        The RMS of the Gaussian parameter initialization.
    solver : str, optional
        The solver the use for optimization.
    solver_kwargs : tuple of key-value pairs, optional
        An optional tuple of tuples of keyword arguments to pass to the solver.
    atol : float, optional
        The absolute tolerance for convergence if `batch_size` is None.
    rtol : float, optional
        The relative tolerance for convergence if `batch_size` is None.
    max_iter : int, optional
        the maximum number of steps to take if `batch_size` is None.
    backend : str, optional
        The computational backend to use. Only "mlx" is currently available.

    Attributes
    ----------
    classes_ : array
        Class labels from the data.
    n_classes_ : int
        Number of unique class labels from the data.
    params_ : tuple of arrays
        The parameters (w0, w, vmat). Only present after fitting.
    converged_ : bool
        Set to True if `batch_size` is None and the fit converged. False
        otherwise.
    """

    json_attributes_ = json_attributes_ = ("_is_fit",
                                           "_rng",
                                           "_mlx_rng_key",
                                           "classes_",
                                           "n_classes_",
                                           "params_",
                                           "converged_",
                                           "n_iter_",
                                           "_label_encoder")

    def __init__(
        self,
        rank=8,
        random_state=None,
        batch_size=None,
        lambda_v=0,
        lambda_w=0,
        init_scale=0.1,
        solver="Lion",
        solver_kwargs=(("learning_rate", 1e-2),),
        atol=1e-4,
        rtol=1e-4,
        max_iter=1000,
        backend="mlx",
    ):
        self.rank = rank
        self.random_state = random_state
        self.batch_size = batch_size
        self.lambda_v = lambda_v
        self.lambda_w = lambda_w
        self.init_scale = init_scale
        self.solver = solver
        self.solver_kwargs = solver_kwargs
        self.atol = atol
        self.rtol = rtol
        self.max_iter = max_iter
        self.backend = backend

    def fit(self, X, y):
        """Fit the FM to data `X` and `y`.

        Parameters
        ----------
        X : array-like
            An array of shape `(n_samples, n_features)`.
        y : array-like
            An array of labels of shape `(n_samples)`.

        Returns
        -------
        self : object
            The fit estimator.
        """

        self._is_fit = False
        return self._partial_fit(X, y, n_epochs=self.max_iter)

    def partial_fit(self, X, y, classes=None):
        """Fit the FM to data `X` and `y` for a single epoch.

        Parameters
        ----------
        X : array-like
            An array of shape `(n_samples, n_features)`.
        y : array-like
            An array of labels of shape `(n_samples)`.
        classes : array-like, optional
            If given, an optional array of unique class labels
            that is used instead of extracting them from the input
            `y`.


        Returns
        -------
        self : object
            The fit estimator.
        """
        return self._partial_fit(X, y, classes=classes)

    def _init_numpy(self, X, y, classes=None):
        X, y = validate_data(self, X=X, y=y, reset=True)

        tot = type_of_target(y, raise_unknown=True)
        if tot not in ["binary", "multiclass"]:
            raise ValueError(
                "Class labels `y` are not the right kind "
                f"of target! Got '{tot}' for '{y}'."
            )

        if classes is not None:
            self._label_encoder = _LabelEncoder().fit(classes)
        else:
            self._label_encoder = _LabelEncoder().fit(y)
        self.classes_ = self._label_encoder.classes_
        self.n_classes_ = len(self.classes_)
        return X, y

    def _init_mlx(self, X, y, classes=None):
        y = mx.round(y).astype(mx.int32)
        if classes is not None:
            self.classes_ = mx.unique(mx.round(classes).astype(mx.int32))
        else:
            self.classes_ = mx.unique(y)
        self.n_classes_ = len(self.classes_)

        validate_data(
            self,
            X=np.ones((1, X.shape[1])),
            y=np.ones(1, dtype=np.int32),
            reset=True,
        )

        if not mx.array_equal(mx.arange(self.n_classes_), self.classes_):
            raise ValueError(
                "For MXNet array inputs, the classes must be integers "
                "from 0 to n_classes_ - 1!"
            )

        return X, y

    def _init_from_json(self, X=None, y=None, classes=None, **kwargs):
        self.n_iter_ = kwargs.get("n_iter_", 0)
        self._rng = kwargs.get("_rng", check_random_state(self.random_state))
        if "_mlx_rng_key" in kwargs:
            self._mlx_rng_key = kwargs["_mlx_rng_key"]
        else:
            self._mlx_rng_key = mx.random.key(
                self._rng.randint(low=1, high=int(2**31))
            )
        self.converged_ = kwargs.get(
            "converged_",
            False,
        )
        self._is_fit = kwargs.get("_is_fit", True)

        if X is None and y is None:
            # restore strictly from JSON
            if "classes_" in kwargs:
                self.classes_ = kwargs["classes_"]
            if "n_classes_" in kwargs:
                self.n_classes_ = kwargs["n_classes_"]
            if "_label_encoder" in kwargs:
                self._label_encoder = kwargs["_label_encoder"]
        else:
            if isinstance(X, mx.array) and isinstance(y, mx.array):
                X, y = self._init_mlx(X, y, classes=classes)
            else:
                X, y = self._init_numpy(X, y, classes=classes)

        if "params_" not in kwargs:
            self._mlx_rng_key, subkey = mx.random.split(self._mlx_rng_key)
            w0 = self.init_scale * mx.random.normal(shape=(self.n_classes_,),
                                                    key=subkey)
            self._mlx_rng_key, subkey = mx.random.split(self._mlx_rng_key)
            w = self.init_scale * mx.random.normal(shape=(self.n_features_in_,
                                                          self.n_classes_,),
                                                   key=subkey
                                                   )
            self._mlx_rng_key, subkey = mx.random.split(self._mlx_rng_key)
            vmat = self.init_scale * mx.random.normal(
                shape=(self.n_features_in_, self.rank, self.n_classes_), key=subkey
            )
            self.params_ = {
                "w0": w0,
                "w": w,
                "vmat": vmat
            }
        else:
            self.params_ = kwargs["params_"]

        return X, y

    def _partial_fit(self, X, y, classes=None, n_epochs=1):
        was_fit = getattr(self, "_is_fit", False)
        if not was_fit:
            X, y = self._init_from_json(X=X, y=y, classes=classes)
            self.loss_history_ = []
            if not hasattr(self, "_optimizer"):
                kwargs = dict(self.solver_kwargs or tuple())
                opt_class = getattr(optim, self.solver, None)
                if opt_class is None:
                    raise ValueError(
                        f"Unknown solver {self.solver!r}. Available:\
                              {[n for n in dir(optim) if not n.startswith('_')]}"
                    )
                if isinstance(opt_class, type):
                    self._optimizer = opt_class(**kwargs)
                else:
                    self._optimizer = opt_class
        else:
            if isinstance(X, mx.array):
                y = mx.round(y).astype(mx.int32)
            else:
                # Fallback
                y = mx.array(self._label_encoder.transform(y))
                X = mx.array(X)

        for _ in range(n_epochs):
            #prev_params = self.params_

            if self.batch_size is not None:
                self._mlx_rng_key, subkey = mx.random.split(self._mlx_rng_key)
                inds = mx.random.permutation(subkey, X.shape[0])
                X = X[inds, :]
                y = y[inds]
                for start in range(0, X.shape[0], self.batch_size):
                    end = min(start + self.batch_size, X.shape[0])
                    Xb = X[start:end, :]
                    yb = y[start:end]

                    new_value, grads = _value_and_grad_mlx_loss_func(
                        self.params_, Xb, yb, self.lambda_v, self.lambda_w
                    )
                    self.loss_history_.append(new_value)

                    self.params_ = self._optimizer.apply_gradients(grads, self.params_)
                    mx.eval(self.params_, self._optimizer.state, new_value)
            else:
                X = mx.array(X)
                y = mx.array(y)
                new_value, grads = _value_and_grad_mlx_loss_func(
                    self.params_, X, y, self.lambda_v, self.lambda_w
                )
                self.loss_history_.append(new_value)

                self.params_ = self._optimizer.apply_gradients(grads, self.params_)
                mx.eval(self.params_, self._optimizer.state, new_value)

            self.n_iter_ += 1

            # if self.batch_size is not None:
            #     if self.n_iter_ > 1 and (
            #         mx.all(
            #             mx.array([
            #                 mx.allclose(new_p, p, atol=self.atol, rtol=self.rtol)
            #                 for new_p, p in zip(self.params_.values(),
            #                                     prev_params.values())
            #             ])
            #         )
            #     ):
            #         self.converged_ = True
            #         break

        # self.loss_history_ = [
        #     float(loss) if isinstance(loss, mx.array) else loss
        #     for loss in self.loss_history_
        # ]

        self._is_fit = True
        return self

    def predict_log_proba(self, X):
        """Predict the log-probability of each class for data `X`.

        Parameters
        ----------
        X : array-like
            An array of shape `(n_samples, n_features)`.

        Returns
        -------
        log_proba : array-like
            An array of labels of shape `(n_samples, n_classes_)` if `n_classes_` > 2,
            else `(n_samples)`.
        """

        if not isinstance(X, (mx.array)):
            X = validate_data(self, X=X, reset=False)
            X = mx.array(X)

        if not getattr(self, "_is_fit", False):
            raise NotFittedError(
                "FMClassifier must be fit before calling `predict_log_proba`!"
            )
        return _call_in_batches_maybe(self, _mlx_log_proba, X)

    def predict_proba(self, X):
        """Predict the probability of each class for data `X`.

        Parameters
        ----------
        X : array-like
            An array of shape `(n_samples, n_features)`.

        Returns
        -------
        proba : array-like
            An array of labels of shape `(n_samples, n_classes_)` if `n_classes_` > 2,
            else `(n_samples)`.
        """

        if not isinstance(X, (mx.array)):
            X = validate_data(self, X=X, reset=False)
            X = mx.array(X)
        if not getattr(self, "_is_fit", False):
            raise NotFittedError(
                "FMClassifier must be fit before calling `predict_proba`!"
            )
        return _call_in_batches_maybe(self, _mlx_proba, X)

    def predict(self, X):
        """Predict the class for data `X`.

        Parameters
        ----------
        X : array-like
            An array of shape `(n_samples, n_features)`.

        Returns
        -------
        y : array-like
            An array of labels of shape `(n_samples)`.
        """

        if not isinstance(X, (mx.array)):
            X = validate_data(self, X=X, reset=False)
            X = mx.array(X)
        if not getattr(self, "_is_fit", False):
            raise NotFittedError("FMClassifier must be fit before calling `predict`!")

        retval = _call_in_batches_maybe(self, _mlx_predict, X)

        retval = np.array(retval)
        if hasattr(self, "_label_encoder"):
            retval = self._label_encoder.inverse_transform(retval)
        return retval

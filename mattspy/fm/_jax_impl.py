import jax
import optax
from jax import numpy as jnp
from jax.tree_util import Partial as partial

import numpy as np
from optax.losses import softmax_cross_entropy_with_integer_labels
from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.utils import check_random_state
from sklearn.utils.validation import validate_data
from sklearn.utils.multiclass import type_of_target
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import LabelEncoder

from mattspy.json import EstimatorToFromJSONMixin


@jax.jit
def _lowrank_twoway_term(x, vmat):
    fterm = jnp.einsum("np,pk...->nk...", x, vmat)
    sterm = jnp.einsum("np,pk...->nk...", x**2, vmat**2)
    return 0.5 * jnp.sum(fterm**2 - sterm, axis=1)


@jax.jit
def _fm_eval(x, w0, w, vmat):
    return w0 + jnp.einsum("np,p...->n...", x, w) + _lowrank_twoway_term(x, vmat)


@partial(jax.jit, static_argnames=("n_features", "rank", "n_classes"))
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


@jax.jit
def _combine_fm_params(w0, w, vmat):
    return jnp.concatenate([jnp.atleast_1d(w0).flatten(), w.flatten(), vmat.flatten()])


@jax.jit
def _jax_logits(params, X):
    w0, w, vmat = params
    logits = _fm_eval(X, w0, w, vmat)
    return logits


@jax.jit
def _jax_log_proba(params, X):
    return jax.nn.log_softmax(
        _jax_logits(params, X),
        axis=-1,
    )


@jax.jit
def _jax_proba(params, X):
    return jax.nn.softmax(
        _jax_logits(params, X),
        axis=-1,
    )


@jax.jit
def _jax_predict(params, X):
    return jnp.argmax(
        _jax_logits(params, X),
        axis=-1,
    )


@jax.jit
def _jax_loss_func(params, X, y, lambda_v, lambda_w):
    w0, w, vmat = params
    logits = _fm_eval(X, w0, w, vmat)
    loss = jnp.mean(softmax_cross_entropy_with_integer_labels(logits, y, axis=-1))
    loss = jax.lax.cond(
        lambda_v > 0,
        lambda loss: loss + lambda_v * jnp.sum(vmat**2),
        lambda loss: loss,
        loss,
    )
    loss = jax.lax.cond(
        lambda_w > 0,
        lambda loss: loss + lambda_w * jnp.sum(w**2),
        lambda loss: loss,
        loss,
    )
    return loss


_value_and_grad_from_state_jax_loss_func = jax.jit(
    optax.value_and_grad_from_state(_jax_loss_func),
)
_grad_jax_loss_func = jax.jit(
    jax.grad(_jax_loss_func),
)
_value_and_grad_jax_loss_func = jax.jit(
    jax.value_and_grad(_jax_loss_func),
)


def _call_in_batches_maybe(self, func, X):
    if self.batch_size is not None:
        vals = []
        for start in range(0, X.shape[0], self.batch_size):
            end = min(start + self.batch_size, X.shape[0])
            Xb = X[start:end, :]
            vals.append(func(self.params_, Xb))
        return jnp.concatenate(vals, axis=0)
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
        The solver the use from the `optax` package.
    solver_kwargs : tuple of key-value pairs, optional
        An optional tuple of tuples of keyword arguments to pass to the solver.
    atol : float, optional
        The absolute tolerance for convergence if `batch_size` is None.
    rtol : float, optional
        The relative tolerance for convergence if `batch_size` is None.
    max_iter : int, optional
        the maximum number of steps to take if `batch_size` is None.
    backend : str, optional
        The computational backend to use. Only "jax" is currently available.

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

    json_attributes_ = (
        "_is_fit",
        "_rng",
        "_jax_rng_key",
        "classes_",
        "n_classes_",
        "params_",
        "converged_",
        "n_iter_",
        "_label_encoder",
    )

    def __init__(
        self,
        rank=8,
        random_state=None,
        batch_size=None,
        lambda_v=0,
        lambda_w=0,
        init_scale=0.1,
        solver="lion",
        solver_kwargs=(("learning_rate", 1e-2),),
        atol=1e-4,
        rtol=1e-4,
        max_iter=1000,
        backend="jax",
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
        return self._partial_fit(self.max_iter, X, y)

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
        return self._partial_fit(1, X, y, classes=classes)

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

    def _init_jax(self, X, y, classes=None):
        y = jnp.rint(y).astype(jnp.int32)
        if classes is not None:
            self.classes_ = jnp.unique(jnp.rint(classes).astype(jnp.int32))
        else:
            self.classes_ = jnp.unique(y)
        self.n_classes_ = len(self.classes_)

        validate_data(
            self,
            X=np.ones((1, X.shape[1])),
            y=np.ones(1, dtype=np.int32),
            reset=True,
        )

        if not jnp.array_equal(jnp.arange(self.n_classes_), self.classes_):
            raise ValueError(
                "For JAX array inputs, the classes must be integers "
                "from 0 to n_classes_ - 1!"
            )

        return X, y

    def _init_from_json(self, X=None, y=None, classes=None, **kwargs):
        self.n_iter_ = kwargs.get("n_iter_", 0)
        self._rng = kwargs.get("_rng", check_random_state(self.random_state))
        if "_jax_rng_key" in kwargs:
            self._jax_rng_key = kwargs["_jax_rng_key"]
        else:
            self._jax_rng_key = jax.random.key(
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
            if not (isinstance(X, jnp.ndarray) and isinstance(y, jnp.ndarray)):
                X, y = self._init_numpy(X, y, classes=classes)
            else:
                X, y = self._init_jax(X, y, classes=classes)

        if "params_" not in kwargs:
            self._jax_rng_key, subkey = jax.random.split(self._jax_rng_key)
            w0 = jax.random.normal(subkey, shape=(self.n_classes_))
            self._jax_rng_key, subkey = jax.random.split(self._jax_rng_key)
            w = jax.random.normal(subkey, shape=(self.n_features_in_, self.n_classes_))
            self._jax_rng_key, subkey = jax.random.split(self._jax_rng_key)
            vmat = jax.random.normal(
                subkey, shape=(self.n_features_in_, self.rank, self.n_classes_)
            )
            self.params_ = (w0, w, vmat)
        else:
            self.params_ = kwargs["params_"]

        return X, y

    def _partial_fit(self, n_epochs, X, y, classes=None):
        was_fit = getattr(self, "_is_fit", False)
        if not was_fit:
            X, y = self._init_from_json(X=X, y=y, classes=classes)
            self.loss_history_ = []
        else:
            if not (isinstance(X, jnp.ndarray) and isinstance(y, jnp.ndarray)):
                X, y = validate_data(self, X=X, y=y, reset=False)
            else:
                y = jnp.rint(y).astype(jnp.int32)

        if not (isinstance(X, jnp.ndarray) and isinstance(y, jnp.ndarray)):
            y = self._label_encoder.transform(y)
            X = jnp.array(X)
            y = jnp.array(y)

        kwargs = {k: v for k, v in (self.solver_kwargs or tuple())}
        if not was_fit:
            # initialize optimizer only if first call to partial fit
            optimizer = getattr(optax, self.solver)(**kwargs)
            self._optimizer = optimizer
        else:
            optimizer = self._optimizer
        if not was_fit:
            # initialize opt_state only if first call to partial fit
            self._opt_state = self._optimizer.init(self.params_)
            opt_state = self._opt_state
        else:
            opt_state = self._opt_state

        new_value = None

        for _ in range(n_epochs):
            value = new_value

            if self.solver not in ["lbfgs"]:
                if self.batch_size is not None:
                    self._jax_rng_key, subkey = jax.random.split(self._jax_rng_key)
                    inds = jax.random.permutation(subkey, X.shape[0])
                    for start in range(0, X.shape[0], self.batch_size):
                        end = min(start + self.batch_size, X.shape[0])
                        Xb = X[inds[start:end], :]
                        yb = y[inds[start:end]]
                        new_value, grads = _value_and_grad_jax_loss_func(
                            self.params_, Xb, yb, self.lambda_v, self.lambda_w
                            )
                        self.loss_history_.append(new_value)
                        updates, opt_state = optimizer.update(
                            grads, opt_state, self.params_
                        )
                        new_params = optax.apply_updates(self.params_, updates)
                        self.params_ = new_params
                else:
                    new_value, grads = _value_and_grad_jax_loss_func(
                        self.params_, X, y, self.lambda_v, self.lambda_w
                    )
                    self.loss_history_.append(new_value)
                    updates, opt_state = optimizer.update(
                        grads, opt_state, self.params_
                    )
                    new_params = optax.apply_updates(self.params_, updates)
            else:
                new_value, grads = _value_and_grad_from_state_jax_loss_func(
                    self.params_,
                    X,
                    y,
                    self.lambda_v,
                    self.lambda_w,
                    state=opt_state,
                )
                self.loss_history_.append(new_value)
                updates, opt_state = optimizer.update(
                    grads,
                    opt_state,
                    self.params_,
                    value=new_value,
                    grad=grads,
                    value_fn=partial(
                        _jax_loss_func,
                        X=X,
                        y=y,
                        lambda_v=self.lambda_v,
                        lambda_w=self.lambda_w,
                    ),
                )
                new_params = optax.apply_updates(self.params_, updates)

            self.n_iter_ += 1
            if self.n_iter_ > 1 and (
                all(
                    [
                        jnp.allclose(new_p, p, atol=self.atol, rtol=self.rtol)
                        for new_p, p in zip(new_params, self.params_)
                    ]
                )
                or (
                    self.solver in ["lbfgs"]
                    and jnp.allclose(value, new_value, atol=self.atol, rtol=self.rtol)
                )
            ):
                self.converged_ = True
                break

            self.params_ = new_params

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

        if not isinstance(X, jnp.ndarray):
            X = validate_data(self, X=X, reset=False)
        if not getattr(self, "_is_fit", False):
            raise NotFittedError(
                "FMClassifier must be fit before calling `predict_log_proba`!"
            )
        return _call_in_batches_maybe(self, _jax_log_proba, X)

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

        if not isinstance(X, jnp.ndarray):
            X = validate_data(self, X=X, reset=False)
        if not getattr(self, "_is_fit", False):
            raise NotFittedError(
                "FMClassifier must be fit before calling `predict_proba`!"
            )
        return _call_in_batches_maybe(self, _jax_proba, X)

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

        if not isinstance(X, jnp.ndarray):
            X = validate_data(self, X=X, reset=False)
        if not getattr(self, "_is_fit", False):
            raise NotFittedError("FMClassifier must be fit before calling `predict`!")

        retval = _call_in_batches_maybe(self, _jax_predict, X)
        if hasattr(self, "_label_encoder"):
            retval = self._label_encoder.inverse_transform(retval)
        return retval

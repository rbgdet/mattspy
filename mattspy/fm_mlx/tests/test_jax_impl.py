import pickle
from io import BytesIO

import numpy as np
import jax.numpy as jnp
import pytest
from sklearn.utils.estimator_checks import check_estimator
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import cross_val_predict, KFold


from mattspy.json import dumps, loads
from mattspy.fm import FMClassifier
from mattspy.fm._jax_impl import (
    _lowrank_twoway_term,
    _fm_eval,
    _extract_fm_params,
    _combine_fm_params,
    _LabelEncoder,
)

RANDOM_SEED = 42


def _gen_test_data(n_samples, n_features, rank, flatten=False, seed=42, n_classes=None):
    rng = np.random.default_rng(seed=seed)
    x = rng.normal(size=(n_samples, n_features))
    if flatten:
        x = x.flatten()

    if n_classes is None:
        vmat = rng.normal(size=(n_features, rank))
        w = rng.normal(size=(n_features))
        w0 = rng.normal()
    else:
        vmat = rng.normal(size=(n_features, rank, n_classes))
        w = rng.normal(size=(n_features, n_classes))
        w0 = rng.normal(size=n_classes)

    return {"x": x, "vmat": vmat, "w": w, "w0": w0}


@pytest.mark.parametrize("n_samples", (1, 10))
def test_fm_lowrank_twoway_term(n_samples):
    n_features = 13
    rank = 4
    data = _gen_test_data(n_samples, n_features, rank, flatten=False)
    x = data["x"]
    vmat = data["vmat"]

    vals = _lowrank_twoway_term(x, vmat)
    true_vals = []
    for k in range(x.shape[0]):
        _val = 0.0
        for i in range(x.shape[1]):
            for j in range(i + 1, x.shape[1]):
                _val += np.sum(vmat[i, :] * vmat[j, :]) * x[k, i] * x[k, j]
        true_vals.append(_val)

    assert vals.shape == x.shape[:1]
    np.testing.assert_allclose(vals, true_vals, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("n_samples", (1, 10))
@pytest.mark.parametrize("n_classes", (1, 7))
def test_fm_lowrank_twoway_term_classes(n_samples, n_classes):
    n_features = 13
    rank = 4
    data = _gen_test_data(
        n_samples, n_features, rank, flatten=False, n_classes=n_classes
    )
    x = data["x"]
    vmat = data["vmat"]

    vals = _lowrank_twoway_term(x, vmat)
    true_vals = []
    for k in range(x.shape[0]):
        _val = 0.0
        for i in range(x.shape[1]):
            for j in range(i + 1, x.shape[1]):
                _val += np.sum(vmat[i, ...] * vmat[j, ...], axis=0) * x[k, i] * x[k, j]
        true_vals.append(_val)

    assert vals.shape == (x.shape[0], n_classes)
    np.testing.assert_allclose(vals, true_vals, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("n_samples", (1, 10))
def test_fm_eval(n_samples):
    n_features = 13
    rank = 4
    data = _gen_test_data(n_samples, n_features, rank, flatten=False)
    x = data["x"]
    vmat = data["vmat"]
    w = data["w"]
    w0 = data["w0"]

    vals = _fm_eval(x, w0, w, vmat)
    true_vals = []
    for k in range(x.shape[0]):
        _val = w0
        _val += np.sum(x[k, :] * w)
        for i in range(x.shape[1]):
            for j in range(i + 1, x.shape[1]):
                _val += np.sum(vmat[i, :] * vmat[j, :]) * x[k, i] * x[k, j]
        true_vals.append(_val)

    assert vals.shape == x.shape[:1]
    np.testing.assert_allclose(vals, true_vals, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("n_samples", (1, 10))
@pytest.mark.parametrize("n_classes", (1, 7))
def test_fm_eval_classes(n_samples, n_classes):
    n_features = 13
    rank = 4
    data = _gen_test_data(
        n_samples, n_features, rank, flatten=False, n_classes=n_classes
    )
    x = data["x"]
    vmat = data["vmat"]
    w = data["w"]
    w0 = data["w0"]

    vals = _fm_eval(x, w0, w, vmat)
    true_vals = []
    for k in range(x.shape[0]):
        _val = w0.copy()
        _val += np.sum(x[k, :].reshape(-1, 1) * w, axis=0)
        for i in range(x.shape[1]):
            for j in range(i + 1, x.shape[1]):
                _val += np.sum(vmat[i, ...] * vmat[j, ...], axis=0) * x[k, i] * x[k, j]
        true_vals.append(_val)

    assert vals.shape == (x.shape[0], n_classes)
    np.testing.assert_allclose(vals, true_vals, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("n_classes", [-1, 0, None, 1, 7])
@pytest.mark.parametrize("n_features", [1, 3, 11])
@pytest.mark.parametrize("rank", [1, 3])
def test_fm_extract_combine_fm_params(n_features, rank, n_classes):
    data = _gen_test_data(
        1,
        n_features,
        rank,
        n_classes=None if n_classes is None or n_classes < 1 else n_classes,
    )
    w0 = data["w0"]
    w = data["w"]
    vmat = data["vmat"]

    params = _combine_fm_params(w0, w, vmat)
    if n_classes is None or n_classes < 1:
        np.testing.assert_allclose(params[0], w0)
        np.testing.assert_allclose(params[1 : 1 + n_features], w)
        np.testing.assert_allclose(params[1 + n_features :], vmat.flatten())
    else:
        np.testing.assert_allclose(params[:n_classes], w0)
        np.testing.assert_allclose(
            params[n_classes : n_classes + n_features * n_classes], w.flatten()
        )
        np.testing.assert_allclose(
            params[n_classes + n_features * n_classes :], vmat.flatten()
        )

    tw0, tw, tvmat = _extract_fm_params(params, n_features, rank, n_classes)
    np.testing.assert_allclose(tw0, w0)
    np.testing.assert_allclose(tw, w)
    np.testing.assert_allclose(tvmat, vmat)


def test_fm_check_estimator():
    check_estimator(
        FMClassifier(random_state=RANDOM_SEED),
    )


def test_fm_partial_fit():
    X, y = load_iris(return_X_y=True)
    X = StandardScaler().fit_transform(X)

    clf = FMClassifier(batch_size=32, random_state=RANDOM_SEED)
    clf.partial_fit(X, y)
    init_auc = roc_auc_score(y, clf.predict_proba(X), multi_class="ovo")
    for _ in range(20):
        clf.partial_fit(X, y)

    final_auc = roc_auc_score(y, clf.predict_proba(X), multi_class="ovo")
    assert final_auc > init_auc
    assert final_auc > 0.90

    assert not clf.converged_


def test_fm_partial_fit_classes():
    X, y = load_iris(return_X_y=True)
    X = StandardScaler().fit_transform(X)

    clf = FMClassifier(batch_size=32, random_state=RANDOM_SEED)
    clf.partial_fit(X[:20], y[:20], classes=np.unique(y))
    init_auc = roc_auc_score(y, clf.predict_proba(X), multi_class="ovo")
    for _ in range(20):
        clf.partial_fit(X, y)

    final_auc = roc_auc_score(y, clf.predict_proba(X), multi_class="ovo")
    assert final_auc > init_auc
    assert final_auc > 0.90


def test_fm_partial_fit_classes_raises():
    with pytest.raises(ValueError):
        X, y = load_iris(return_X_y=True)
        X = StandardScaler().fit_transform(X)

        clf = FMClassifier(batch_size=32, random_state=RANDOM_SEED)
        clf.partial_fit(X, y, classes=np.array([0, 3, 4]))


def test_fm_output_shapes():
    X, y = load_iris(return_X_y=True)
    X = StandardScaler().fit_transform(X)

    clf = FMClassifier(random_state=RANDOM_SEED)
    clf.fit(X, y)
    assert np.array_equal(clf.classes_, np.unique(y))
    final_auc = roc_auc_score(y, clf.predict_proba(X), multi_class="ovo")
    assert final_auc > 0.90

    assert clf.predict_proba(X).shape == (X.shape[0], clf.n_classes_)
    assert clf.predict_log_proba(X).shape == (X.shape[0], clf.n_classes_)
    assert clf.predict(X).shape == (X.shape[0],)

    assert clf.predict_proba(X[:1]).shape == (1, clf.n_classes_)
    assert clf.predict_log_proba(X[:1]).shape == (1, clf.n_classes_)
    assert clf.predict(X[:1]).shape == (1,)

    clf.fit(X, (y == y[0]).astype(int))
    assert np.array_equal(clf.classes_, [0, 1])
    final_auc = roc_auc_score((y == y[0]).astype(int), clf.predict_proba(X)[:, 1])
    assert final_auc > 0.90

    assert clf.predict_proba(X).shape == (X.shape[0], clf.n_classes_)
    assert clf.predict_log_proba(X).shape == (X.shape[0], clf.n_classes_)
    assert clf.predict(X).shape == (X.shape[0],)

    assert clf.predict_proba(X[:1]).shape == (1, clf.n_classes_)
    assert clf.predict_log_proba(X[:1]).shape == (1, clf.n_classes_)
    assert clf.predict(X[:1]).shape == (1,)


def test_fm_cross_val():
    X, y = load_iris(return_X_y=True)
    X = StandardScaler().fit_transform(X)
    clf = FMClassifier(random_state=RANDOM_SEED)
    cv = KFold(n_splits=4, random_state=457, shuffle=True)
    proba_oos = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")
    final_auc = roc_auc_score(y, proba_oos, multi_class="ovo")
    assert final_auc > 0.90


def test_fm_jax_arrays():
    X, y = load_iris(return_X_y=True)
    X = jnp.array(StandardScaler().fit_transform(X))
    y = jnp.array(LabelEncoder().fit_transform(y))

    for _y in [y, y.astype(float) + 0.1, y.astype(float)]:
        clf = FMClassifier(rtol=1e-5, atol=1e-5, random_state=RANDOM_SEED)
        clf.fit(X, _y)
        assert clf.n_features_in_ == X.shape[1]
        assert clf.n_classes_ == len(jnp.unique(y))
        final_auc = roc_auc_score(y, clf.predict_proba(X), multi_class="ovo")
        assert final_auc > 0.90
        final_auc = roc_auc_score(
            y, jnp.exp(clf.predict_log_proba(X)), multi_class="ovo"
        )
        assert final_auc > 0.90
        final_acc = accuracy_score(y, clf.predict(X))
        assert final_acc > 0.90

    clf = FMClassifier(batch_size=32, random_state=RANDOM_SEED)
    clf.partial_fit(X, y)
    assert clf.n_features_in_ == X.shape[1]
    assert clf.n_classes_ == len(jnp.unique(y))
    init_auc = roc_auc_score(y, clf.predict_proba(X), multi_class="ovo")
    for _ in range(20):
        clf.partial_fit(X, y)
    final_auc = roc_auc_score(y, clf.predict_proba(X), multi_class="ovo")
    assert final_auc > init_auc
    assert final_auc > 0.90

    clf = FMClassifier(batch_size=32, random_state=RANDOM_SEED)
    clf.partial_fit(X[:20], y[:20], classes=jnp.unique(y))
    assert clf.n_features_in_ == X.shape[1]
    assert clf.n_classes_ == len(jnp.unique(y))
    init_auc = roc_auc_score(y, clf.predict_proba(X), multi_class="ovo")
    for _ in range(20):
        clf.partial_fit(X, y)
    final_auc = roc_auc_score(y, clf.predict_proba(X), multi_class="ovo")
    assert final_auc > init_auc
    assert final_auc > 0.90

    with pytest.raises(ValueError):
        clf = FMClassifier(rtol=1e-5, atol=1e-5, random_state=RANDOM_SEED)
        clf.fit(X, y + 1)

    with pytest.raises(ValueError):
        clf = FMClassifier(rtol=1e-5, atol=1e-5, random_state=RANDOM_SEED)
        clf.partial_fit(X, y, classes=jnp.unique(y) + 1)


def test_fm_pickling_jax():
    X, y = load_iris(return_X_y=True)
    X = jnp.array(StandardScaler().fit_transform(X))
    y = jnp.array(LabelEncoder().fit_transform(y))
    clf = FMClassifier(random_state=RANDOM_SEED)
    clf.fit(X, y)
    probs = clf.predict_proba(X)
    b = BytesIO()
    pickle.dump(clf, b)
    clf_pickled = pickle.loads(b.getvalue())
    probs_pickled = clf_pickled.predict_proba(X)
    assert jnp.allclose(probs, probs_pickled)


def test_fm_pickling():
    X, y = load_iris(return_X_y=True)
    X = StandardScaler().fit_transform(X)
    clf = FMClassifier(random_state=RANDOM_SEED)
    clf.fit(X, y)
    probs = clf.predict_proba(X)
    b = BytesIO()
    pickle.dump(clf, b)
    clf_pickled = pickle.loads(b.getvalue())
    probs_pickled = clf_pickled.predict_proba(X)
    assert np.allclose(probs, probs_pickled)


def test_fm_random_state_handling():
    X, y = load_iris(return_X_y=True)
    X = StandardScaler().fit_transform(X)
    clf = FMClassifier(random_state=RANDOM_SEED)
    probs = clf.fit(X, y).predict_proba(X)
    probs_again = clf.fit(X, y).predict_proba(X)
    assert np.allclose(probs, probs_again)


def test_fm_random_state_handling_jax():
    X, y = load_iris(return_X_y=True)
    X = jnp.array(StandardScaler().fit_transform(X))
    y = jnp.array(LabelEncoder().fit_transform(y))
    clf = FMClassifier(random_state=RANDOM_SEED)
    probs = clf.fit(X, y).predict_proba(X)
    probs_again = clf.fit(X, y).predict_proba(X)
    assert jnp.allclose(probs, probs_again)


def test_label_encoder_check_estimator():
    check_estimator(_LabelEncoder())


def test_label_encoder_to_from_json():
    enc = _LabelEncoder()
    enc.fit(["a", "b", "b", "c"])
    assert set(enc.classes_) == set(["a", "b", "c"])
    new_enc = _LabelEncoder.from_json(enc.to_json())
    assert set(new_enc.classes_) == set(["a", "b", "c"])
    assert np.array_equal(
        new_enc.transform(["a", "b"]), np.array([0, 1], dtype=np.int32)
    )


def test_label_encoder_dumps_loads():
    enc = _LabelEncoder()
    enc.fit(["a", "b", "b", "c"])
    assert set(enc.classes_) == set(["a", "b", "c"])
    new_enc = dumps(enc)
    new_enc = loads(new_enc)
    assert set(new_enc.classes_) == set(["a", "b", "c"])
    assert np.array_equal(
        new_enc.transform(["a", "b"]), np.array([0, 1], dtype=np.int32)
    )


def test_fm_dumps_loads():
    X, y = load_iris(return_X_y=True)
    X = StandardScaler().fit_transform(X)

    clf = FMClassifier(batch_size=32, random_state=RANDOM_SEED)
    clf.partial_fit(X, y)
    init_auc = roc_auc_score(y, clf.predict_proba(X), multi_class="ovo")

    new_clf = dumps(clf)
    new_clf = loads(new_clf)
    final_auc = roc_auc_score(y, clf.predict_proba(X), multi_class="ovo")

    assert init_auc == final_auc

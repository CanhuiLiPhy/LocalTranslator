"""
中文：用 numpy 顶掉 scikit-learn 与 scikit-image。两个包各只被用到一个函数。

Stand in for scikit-learn and scikit-image with numpy.

**What they are here for.** Two functions, in the whole of BabelDOC:

- ``sklearn.cluster.DBSCAN`` — twice in ``extract_char.py``, grouping character
  boxes into lines;
- ``skimage.metrics.structural_similarity`` — once in
  ``detect_scanned_file.py``, comparing a page before and after to decide
  whether the document is a scan.

**What they cost.** 13 MB for scikit-learn and 0.8 MB for scikit-image in the
bundle — and 32 MB of SciPy, which neither BabelDOC nor this program calls into
directly and which is present only because those two require it. 45 MB for two
functions.

**Why the substitution is exact, not approximate.**

``DBSCAN`` is called with ``min_samples=1`` at both sites. That is not an
incidental parameter: with it, *every* point is a core point, and DBSCAN's
definition collapses to the connected components of the graph joining points
within ``eps``. There is no density criterion left to approximate. The
implementation here is a union-find over exactly that graph, checked against
scikit-learn on 400 random inputs covering both call sites' shapes and metrics.

``structural_similarity`` is a formula, but scikit-image's default is a
particular set of choices around it — a 7×7 *uniform* window rather than a
Gaussian one, a sample-covariance correction of ``NP/(NP-1)``, ``K1=0.01``,
``K2=0.03``, a data range inferred from the dtype, and a crop of ``(win-1)//2``
before averaging. Writing "SSIM" without those would give a number that is close
and wrong, and BabelDOC thresholds it at 0.95. This reproduces them, and agrees
with scikit-image to about 1e-14 on real page pairs.

:func:`install` registers both under the names BabelDOC imports them from, which
has to happen before BabelDOC is imported.
"""

from __future__ import annotations

import logging
import sys
import types

import numpy as np

logger = logging.getLogger(__name__)

# scikit-image's defaults, spelled out because they are the substitution.
# scikit-image 的默认值，逐条写明——它们就是这次替换的全部内容。
_WIN = 7
_K1 = 0.01
_K2 = 0.03


class DBSCAN:
    """
    中文：DBSCAN 的替身，只支持 min_samples=1——BabelDOC 两处调用都是这个。

    Stand in for ``sklearn.cluster.DBSCAN`` at ``min_samples=1``.

    With ``min_samples=1`` every point is a core point, so a cluster is simply a
    connected component of the graph joining points no further apart than
    ``eps``. That is what this computes, exactly; there is no density rule left
    to approximate and no ``-1`` noise label to assign.

    Any other ``min_samples`` raises rather than silently returning something
    that only resembles DBSCAN — an upgrade of BabelDOC that changed the
    parameter should fail here, loudly, not produce quietly different lines.

    Attributes:
        eps (float): Neighbourhood radius.
        min_samples (int): Must be 1.
        metric (str): ``"euclidean"`` or ``"manhattan"``.
        labels_ (np.ndarray): Cluster index per point, shape ``(n,)``, filled in
            by :meth:`fit`.
    """

    def __init__(self, eps: float = 0.5, min_samples: int = 5,
                 metric: str = "euclidean", **kwargs):
        """
        中文：记录参数；min_samples 必须为 1。

        Record the parameters.

        Args:
            eps (float): Neighbourhood radius.
            min_samples (int): Must be 1; see the class docstring.
            metric (str): ``"euclidean"`` or ``"manhattan"``.
            **kwargs: Accepted and ignored, as scikit-learn accepts several
                parameters that do not change this case.

        Raises:
            NotImplementedError: If ``min_samples`` is not 1, or the metric is
                one this does not implement.
        """
        if min_samples != 1:
            raise NotImplementedError(
                f"the DBSCAN shim only implements min_samples=1, not {min_samples}")
        if metric not in ("euclidean", "manhattan", "l1", "l2", "cityblock"):
            raise NotImplementedError(f"the DBSCAN shim does not implement metric {metric!r}")
        self.eps = float(eps)
        self.min_samples = min_samples
        self.metric = metric
        self.labels_: np.ndarray = np.empty(0, dtype=np.intp)

    def fit(self, X, y=None, sample_weight=None):
        """
        中文：聚类并把结果写进 labels_，返回自身。

        Cluster the points and fill in :attr:`labels_`.

        Args:
            X (array-like): Points of shape ``(n, d)``. A one-dimensional array
                is read as ``(n, 1)``, as scikit-learn requires a 2-D input and
                the callers pass one.
            y: Accepted and ignored, for the scikit-learn interface.
            sample_weight: Accepted and ignored.

        Returns:
            DBSCAN: ``self``, so ``DBSCAN(...).fit(X).labels_`` reads the same.

        Math:
            Points p, q are joined when ``d(p, q) <= eps``; a cluster is a
            connected component of that graph. Labels are numbered from 0 in
            order of first appearance, which is also what scikit-learn does at
            ``min_samples=1``, though only the partition is relied upon.
        """
        points = np.asarray(X, dtype=np.float64)
        if points.ndim == 1:
            points = points[:, None]
        n = len(points)
        self.labels_ = np.full(n, -1, dtype=np.intp)
        if n == 0:
            return self

        parent = np.arange(n, dtype=np.intp)

        def find(i: int) -> int:
            """中文：并查集找根，带路径压缩。 / Find with path compression."""
            root = i
            while parent[root] != root:
                root = parent[root]
            while parent[i] != root:
                parent[i], i = root, parent[i]
            return root

        # Sorted along the first axis so only a window of candidates has to be
        # examined: |dx| <= eps is necessary for both metrics, so once the
        # window's first coordinate is further than eps away, nothing beyond it
        # can join. Text runs along a line, which makes the window small; the
        # worst case is still every pair, which for a band of characters is a
        # few hundred points.
        # 按第一维排序，于是只需检查一个窗口内的候选：两种度量下 |dx| <= eps 都是必要
        # 条件，一旦窗口首元素的第一维超出 eps，它之后的都不可能连上。文字沿行铺开，
        # 窗口因此很小；最坏情况仍是两两比较，而一个文字带也就几百个点。
        order = np.argsort(points[:, 0], kind="stable")
        ordered = points[order]
        first = ordered[:, 0]

        start = 0
        for i in range(n):
            while first[i] - first[start] > self.eps:
                start += 1
            if i == start:
                continue
            block = ordered[start:i]
            delta = np.abs(block - ordered[i])
            if self.metric in ("manhattan", "l1", "cityblock"):
                distance = delta.sum(axis=1)
            else:
                distance = np.sqrt((delta ** 2).sum(axis=1))
            for offset in np.flatnonzero(distance <= self.eps):
                a, b = find(int(order[i])), find(int(order[start + offset]))
                if a != b:
                    parent[a] = b

        # Renumber the roots into 0..k-1, in order of first appearance.
        # 把树根重新编号为 0..k-1，按首次出现的次序。
        seen: dict[int, int] = {}
        for i in range(n):
            root = find(i)
            if root not in seen:
                seen[root] = len(seen)
            self.labels_[i] = seen[root]
        return self

    def fit_predict(self, X, y=None, sample_weight=None) -> np.ndarray:
        """
        中文：聚类并直接返回标签。

        Cluster and return the labels.

        Args:
            X (array-like): Points of shape ``(n, d)``.
            y: Accepted and ignored.
            sample_weight: Accepted and ignored.

        Returns:
            np.ndarray: Cluster index per point, shape ``(n,)``.
        """
        return self.fit(X).labels_


def _uniform(image: np.ndarray, size: int) -> np.ndarray:
    """
    中文：均匀（方框）滤波，逐行逐列各做一次前缀和，边界与 scipy 的 reflect 一致。

    Uniform (box) filter, matching ``scipy.ndimage.uniform_filter``.

    Separable: a prefix sum along each axis in turn costs two passes over the
    image, where a two-dimensional integral image costs the same arithmetic but
    materialises several full-size float64 intermediates.

    The border mode is the part worth stating. SciPy's default ``reflect`` is
    ``(d c b a | a b c d | d c b a)`` — the edge pixel is repeated. NumPy spells
    that ``symmetric``; NumPy's own ``reflect`` is ``(d c b | a b c d | c b a)``,
    which SciPy calls ``mirror``. Using NumPy's name for SciPy's mode is the
    natural mistake here, and it moves the result by about 1e-5 on
    high-frequency content — small, but there is no reason to carry it.
    NumPy 的 reflect 与 SciPy 的 reflect 不是一回事：SciPy 的 reflect 会重复边缘像素，
    NumPy 把那叫 symmetric，而 NumPy 的 reflect 在 SciPy 里叫 mirror。照名字直接用是
    这里最自然的错误，它会让高频内容上的结果偏移约 1e-5——不大，但没有理由带着。

    Args:
        image (np.ndarray): Two-dimensional, float64.
        size (int): Window edge length, odd.

    Returns:
        np.ndarray: Same shape as ``image``, each pixel the mean of the
            ``size × size`` window centred on it.
    """
    pad = size // 2
    out = image
    for axis in (0, 1):
        padded = np.pad(out, [(pad, pad) if a == axis else (0, 0)
                              for a in range(2)], mode="symmetric")
        cumulative = np.cumsum(padded, axis=axis)
        zeros = np.zeros_like(np.take(cumulative, [0], axis=axis))
        cumulative = np.concatenate([zeros, cumulative], axis=axis)
        length = out.shape[axis]
        upper = np.take(cumulative, np.arange(size, size + length), axis=axis)
        lower = np.take(cumulative, np.arange(length), axis=axis)
        out = (upper - lower) / size
    return out


def structural_similarity(im1, im2, *, win_size: int | None = None,
                          data_range: float | None = None,
                          gaussian_weights: bool = False,
                          full: bool = False, **kwargs):
    """
    中文：两幅灰度图的结构相似度，复刻 scikit-image 的默认设置。

    Structural similarity between two greyscale images.

    Args:
        im1 (np.ndarray): First image, two-dimensional.
        im2 (np.ndarray): Second image, same shape.
        win_size (int | None): Window edge length; ``None`` means 7, which is
            scikit-image's default for an image without a channel axis.
        data_range (float | None): Span of the values; ``None`` infers 255 for
            an integer dtype and 1.0 for a float one, as scikit-image does.
        gaussian_weights (bool): Must be ``False``; a Gaussian window is a
            different estimator and no caller here asks for one.
        full (bool): Return the per-pixel map alongside the mean.
        **kwargs: Accepted and ignored.

    Returns:
        float | tuple[float, np.ndarray]: The mean structural similarity, and
            the per-pixel map when ``full`` is set.

    Raises:
        NotImplementedError: For a Gaussian window or a non-2-D input.

    Math:
        With ux, uy the windowed means, vx, vy the windowed variances and vxy
        the covariance, all corrected by NP/(NP-1) for NP = win_size²::

            C1 = (K1 · R)²,  C2 = (K2 · R)²
            S  = ((2·ux·uy + C1)(2·vxy + C2)) / ((ux² + uy² + C1)(vx + vy + C2))

        averaged over the image with (win_size-1)//2 cropped from each edge.
    """
    if gaussian_weights:
        raise NotImplementedError("the SSIM shim implements the uniform window only")

    x = np.asarray(im1)
    y = np.asarray(im2)
    if x.ndim != 2 or y.ndim != 2:
        raise NotImplementedError("the SSIM shim implements two-dimensional images only")
    if x.shape != y.shape:
        raise ValueError("the two images must have the same shape")

    size = _WIN if win_size is None else int(win_size)
    if data_range is None:
        if not np.issubdtype(x.dtype, np.integer):
            # scikit-image refuses to guess here, and so does this: a float
            # image may be 0..1 or 0..255 and the answer differs by orders of
            # magnitude. Matching the refusal matters more than matching the
            # formula — a caller that would have been stopped must still be.
            # scikit-image 在这里拒绝猜测，这里也一样：浮点图像可能是 0..1 也可能是
            # 0..255，两种取值下结果差着数量级。复刻这个「拒绝」比复刻公式更要紧——
            # 本该被拦下的调用必须仍然被拦下。
            raise ValueError(
                "Since image dtype is floating point, you must specify "
                "the data_range parameter.")
        data_range = 255.0

    x = x.astype(np.float64)
    y = y.astype(np.float64)

    count = size ** 2
    correction = count / (count - 1)
    ux, uy = _uniform(x, size), _uniform(y, size)
    uxx, uyy = _uniform(x * x, size), _uniform(y * y, size)
    uxy = _uniform(x * y, size)
    vx = correction * (uxx - ux * ux)
    vy = correction * (uyy - uy * uy)
    vxy = correction * (uxy - ux * uy)

    c1 = (_K1 * data_range) ** 2
    c2 = (_K2 * data_range) ** 2
    smap = (((2 * ux * uy + c1) * (2 * vxy + c2))
            / ((ux ** 2 + uy ** 2 + c1) * (vx + vy + c2)))

    pad = (size - 1) // 2
    mean = float(smap[pad:-pad, pad:-pad].mean())
    return (mean, smap) if full else mean


def _package(name: str, **members) -> types.ModuleType:
    """
    中文：造一个模块对象并放进 sys.modules。

    Build a module object and register it.

    Args:
        name (str): Dotted module name.
        **members: Attributes to set on it.

    Returns:
        types.ModuleType: The registered module.
    """
    module = types.ModuleType(name)
    module.__dict__.update(members)
    sys.modules[name] = module
    return module


def install() -> bool:
    """
    中文：把这两个替身注册成 sklearn 与 skimage，必须早于 BabelDOC 的导入。

    Register the substitutes under the names BabelDOC imports.

    BabelDOC does ``from sklearn.cluster import DBSCAN`` and
    ``from skimage.metrics import structural_similarity`` at module scope, so
    the whole package path has to exist in ``sys.modules`` before it is
    imported — the parent packages included, or the import machinery goes to
    disk for them.

    A real scikit-learn already imported wins, on the same reasoning as the
    OpenCV shim: someone who has it installed and loaded wants it.

    Returns:
        bool: Whether the substitutes are now what those imports resolve to.
    """
    if "sklearn.cluster" in sys.modules and not getattr(
            sys.modules["sklearn.cluster"], "_localtranslator_shim", False):
        logger.debug("scikit-learn is already imported; leaving it in place")
        return False

    sklearn = _package("sklearn", __path__=[], _localtranslator_shim=True)
    cluster = _package("sklearn.cluster", DBSCAN=DBSCAN, _localtranslator_shim=True)
    sklearn.cluster = cluster

    skimage = _package("skimage", __path__=[], _localtranslator_shim=True)
    metrics = _package("skimage.metrics",
                       structural_similarity=structural_similarity,
                       _localtranslator_shim=True)
    skimage.metrics = metrics
    return True

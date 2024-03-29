import numpy
from tqdm import tqdm
import math
from feature_vector_projection import FeatureVectorProjection
from kernel_pca import KernelPCA
from kernels import GaussianKernel, GaussianKernelForAngle
from scipy.ndimage import generic_filter
from functools import reduce


class KernelDescriptorsExtractor:
    def __init__(self, gamma_o=5, gamma_c=4, gamma_b=2, gamma_p=3,
                 grid_o_dim=25, grid_c_dims=(5, 5, 5), grid_p_dims=(5, 5),
                 epsilon_g=0.8, epsilon_s=0.2):
        print("basis for orientation")
        k_o = GaussianKernelForAngle(1 / numpy.sqrt(2 * gamma_o))
        self.projector_o = FeatureVectorProjection(k_o)
        X = numpy.linspace(-numpy.pi, numpy.pi, grid_o_dim + 1)[:-1]
        X = X[:, numpy.newaxis]
        self.projector_o.fit(X)

        print("basis for color")
        k_c = GaussianKernel(1 / numpy.sqrt(2 * gamma_c))
        self.projector_c = FeatureVectorProjection(k_c)
        r_step = 1.0 / (grid_c_dims[0] - 1)
        g_step = 1.0 / (grid_c_dims[1] - 1)
        b_step = 1.0 / (grid_c_dims[2] - 1)
        X = numpy.mgrid[0:1 + r_step:r_step, 0:1 +
                        g_step:g_step, 0:1 + b_step:b_step].reshape(3, -1).T
        self.projector_c.fit(X)

        print("basis for binary patterns")
        k_b = GaussianKernel(1 / numpy.sqrt(2 * gamma_b))
        self.projector_b = FeatureVectorProjection(k_b)
        X = numpy.mgrid[0:2:1, 0:2:1, 0:2:1, 0:2:1,
                        0:2:1, 0:2:1, 0:2:1, 0:2:1].reshape(8, -1).T
        self.projector_b.fit(X)

        print("basis for positions")
        k_p = GaussianKernel(1 / numpy.sqrt(2 * gamma_p))
        self.projector_p = FeatureVectorProjection(k_p)
        x_step = 1.0 / (grid_p_dims[0] - 1)
        y_step = 1.0 / (grid_p_dims[1] - 1)
        X = numpy.mgrid[0:1 + x_step:x_step,
                        0:1 + y_step:y_step].reshape(2, -1).T
        self.projector_p.fit(X)

        self.epsilon_g = epsilon_g
        self.epsilon_s = epsilon_s

        kpca_kernel = GaussianKernel(0.4)
        X_p = self.projector_p.predict(self.projector_p.basis)

        kdes_dim = self.projector_o.ndim * self.projector_p.ndim
        X_o = self.projector_o.predict(self.projector_o.basis)
        X_op = numpy.zeros((kdes_dim, kdes_dim))
        for i, (x, y) in enumerate(zip(X_o, X_p)):
            X_op[i, :] = numpy.kron(x, y)
        self.kpca_op = KernelPCA(kpca_kernel)
        self.kpca_op.fit(X_op)

        kdes_dim = self.projector_c.ndim * self.projector_p.ndim
        X_c = self.projector_c.predict(self.projector_c.basis)
        X_cp = numpy.zeros((kdes_dim, kdes_dim))
        pos = 0
        for x in X_c:
            for y in X_p:
                X_cp[pos, :] = numpy.kron(x, y)
                pos += 1
        self.kpca_cp = KernelPCA(kpca_kernel)
        self.kpca_cp.fit(X_cp)

        kdes_dim = self.projector_b.ndim * self.projector_p.ndim
        X_b = self.projector_b.predict(self.projector_b.basis)
        X_bp = numpy.zeros((kdes_dim, kdes_dim))
        pos = 0
        for x in X_b:
            for y in X_p:
                X_bp[pos, :] = numpy.kron(x, y)
                pos += 1
        self.kpca_bp = KernelPCA(kpca_kernel)
        self.kpca_bp.fit(X_bp)

    def _calc_gradient_match_kernel_for_image(self, I, patch_size, subsample):
        nX, nY, nchannels = I.shape

        # precalculate magnitude and angle of gradient in each pixel
        Ig_magnitude = numpy.zeros(I.shape[0:2])
        Ig_angle = numpy.zeros(I.shape[0:2])
        for i in range(nX):
            for j in range(nY):
                chosen_dx, chosen_dy, chosen_magnitude = 0, 0, 0

                for c in range(nchannels):
                    dx, dy = 0, 0
                    if i < nX - 1:
                        dx += I[i + 1, j, c]
                    if i > 0:
                        dx -= I[i - 1, j, c]
                    if j < nY - 1:
                        dy += I[i, j + 1, c]
                    if j > 0:
                        dy -= I[i, j - 1, c]
                    magnitude = dx ** 2 + dy ** 2

                    if magnitude > chosen_magnitude:
                        chosen_magnitude = magnitude
                        chosen_dx = dx
                        chosen_dy = dy

                Ig_magnitude[i, j] = numpy.sqrt(magnitude)
                Ig_angle[i, j] = numpy.arctan2(dx, dy)

        x_step = 1.0 / (patch_size[0] - 1)
        y_step = 1.0 / (patch_size[1] - 1)
        X_p = numpy.mgrid[0:1 + x_step:x_step,
                          0:1 + y_step:y_step].reshape(2, -1).T
        X_p = self.projector_p.predict(X_p)

        patch_x = numpy.arange(patch_size[0]).repeat(patch_size[1])
        patch_y = numpy.tile(numpy.arange(patch_size[1]), patch_size[0])

        kdes_dims = self.projector_o.ndim * self.projector_p.ndim
        ret = numpy.zeros((0, kdes_dims))
        pos = 0
        for sx in range(0, nX - patch_size[0] + 1, subsample[0]):
            for sy in range(0, nY - patch_size[1] + 1, subsample[1]):
                norm = numpy.sum(
                    Ig_magnitude[sx:sx + patch_size[0], sy:sy + patch_size[1]] ** 2)
                norm = numpy.sqrt(self.epsilon_g + norm)

                X_o = Ig_angle[sx:sx + patch_size[0],
                               sy:sy + patch_size[1]].reshape(-1)
                X_o = X_o[:, numpy.newaxis]
                X_o = self.projector_o.predict(X_o)

                aux = numpy.zeros(kdes_dims)
                for x_o, x_p, x, y in zip(X_o, X_p, patch_x, patch_y):
                    aux += Ig_magnitude[x, y] * numpy.kron(x_o, x_p)
                ret = numpy.append(ret, [aux / norm], axis=0)
                pos += 1

        return self.kpca_op.predict(ret).flatten()

    def _calc_shape_match_kernel_for_image(self, I, patch_size, subsample):
        nX, nY, nchannels = I.shape

        # precalculate magnitude and angle of gradient in each pixel
        s_z = generic_filter(I, numpy.nanstd, size=3,
                             mode='constant', cval=numpy.nan).mean(axis=2)

        global result, H, W, pos
        H, W, _ = I.shape
        result = numpy.empty((H, W, (3**2)-1))

        def binary_pattern_extractor(local_mat):
            global result, H, W, pos
            # local_mat[(len(local_mat)//2)+1]
            # print(local_mat, H, W, pos, pos//W, pos % W)
            result[pos//W, pos %
                   W] = numpy.delete(numpy.where(local_mat >= numpy.nanmedian(local_mat), 1, 0), len(local_mat)//2)
            pos += 1
            return 1

        def binary_pattern_extractor_reduce(rest, c):
            global result, H, W, pos
            pos = 0
            generic_filter(I[:, :, c], binary_pattern_extractor,
                           size=3, mode='constant', cval=0)
            return (rest + result) if type(rest) == numpy.ndarray else result

        b = reduce(binary_pattern_extractor_reduce, [0, 1, 2], None) // 3

        x_step = 1.0 / (patch_size[0] - 1)
        y_step = 1.0 / (patch_size[1] - 1)
        X_p = numpy.mgrid[0:1 + x_step:x_step,
                          0:1 + y_step:y_step].reshape(2, -1).T
        X_p = self.projector_p.predict(X_p)

        patch_x = numpy.arange(patch_size[0]).repeat(patch_size[1])
        patch_y = numpy.tile(numpy.arange(patch_size[1]), patch_size[0])

        kdes_dims = self.projector_b.ndim * self.projector_p.ndim
        ret = numpy.zeros((0, kdes_dims))
        pos = 0
        for sx in range(0, nX - patch_size[0] + 1, subsample[0]):
            for sy in range(0, nY - patch_size[1] + 1, subsample[1]):
                norm = numpy.sum(
                    s_z[sx:sx + patch_size[0], sy:sy + patch_size[1]] ** 2)
                norm = numpy.sqrt(self.epsilon_s + norm)

                X_b = b[sx:sx + patch_size[0],
                        sy:sy + patch_size[1]].reshape(8, -1).T
                # X_b = X_b[:, numpy.newaxis]
                X_b = self.projector_b.predict(X_b)

                aux = numpy.zeros(kdes_dims)
                for x_b, x_p, x, y in zip(X_b, X_p, patch_x, patch_y):
                    aux += s_z[x, y] * numpy.kron(x_b, x_p)
                    # res = s_z[x, y] * numpy.kron(x_b, x_p)
                    # aux += numpy.pad(res, (0, (math.ceil(res.shape[0] / kdes_dims) * kdes_dims) - res.shape[0]), constant_values=0).reshape(-1, kdes_dims).T.mean(axis=1).ravel()
                ret = numpy.append(ret, [aux / norm], axis=0)
                pos += 1

        return self.kpca_bp.predict(ret).flatten()

    def _calc_color_match_kernel_for_image(self, I, patch_size, subsample):
        nX, nY, nchannels = I.shape

        x_step = 1.0 / (patch_size[0] - 1)
        y_step = 1.0 / (patch_size[1] - 1)
        X_p = numpy.mgrid[0:1 + x_step:x_step,
                          0:1 + y_step:y_step].reshape(2, -1).T
        X_p = self.projector_p.predict(X_p)

        patch_x = numpy.arange(patch_size[0]).repeat(patch_size[1])
        patch_y = numpy.tile(numpy.arange(patch_size[1]), patch_size[0])

        X_c = numpy.zeros((patch_size[0] * patch_size[1], 3))

        kdes_dims = self.projector_c.ndim * self.projector_p.ndim
        ret = numpy.zeros((0, kdes_dims))
        pos = 0
        for sx in range(0, nX - patch_size[0] + 1, subsample[0]):
            for sy in range(0, nY - patch_size[1] + 1, subsample[1]):
                for i, (x, y) in enumerate(zip(patch_x, patch_y)):
                    X_c[i, :] = I[x, y, :]
                X_c_proj = self.projector_c.predict(X_c)

                aux = numpy.zeros(kdes_dims)
                for x_c, x_p in zip(X_c_proj, X_p):
                    aux += numpy.kron(x_c, x_p)
                ret = numpy.append(ret, [aux], axis=0)
                pos += 1

        return self.kpca_cp.predict(ret, components=200).flatten()

    def predict(self, X, patch_size=(64, 64), subsample=(32, 32), match_kernel='shape'):
        assert X.ndim == 4
        n = X.shape[0]
        print("Match kernel: %s" % match_kernel)

        X_grad = [[]]
        X_color = [[]]
        X_shape = [[]]

        if match_kernel == 'gradient' or match_kernel == 'all':
            X_grad = []
            for i in tqdm(list(range(n))):
                X_grad.append(self._calc_gradient_match_kernel_for_image(
                    X[i, :, :, :], patch_size, subsample))
            X_grad = numpy.array(X_grad)
            # return X_grad
        if match_kernel == 'color' or match_kernel == 'all':
            X_color = []
            for i in tqdm(list(range(n))):
                X_color.append(self._calc_color_match_kernel_for_image(
                    X[i, :, :, :], patch_size, subsample))
            # return X_color
        if match_kernel == 'shape' or match_kernel == 'all':
            X_shape = []
            for i in tqdm(list(range(n))):
                X_shape.append(self._calc_shape_match_kernel_for_image(
                    X[i, :, :, :], patch_size, subsample))
            # return X_shape
        # else:
        #     raise Exception("Unknown match kernel")

        print(numpy.shape(X_grad), numpy.shape(X_color), numpy.shape(X_shape))
        return numpy.concatenate((X_grad, X_color, X_shape), axis=1)

import numpy as np
from numba import njit


GABOR_ORIENTATIONS = 4
GABOR_SIZES = [3, 5]


def make_gabor_bank():
    filters = []
    for size in GABOR_SIZES:
        for theta_idx in range(GABOR_ORIENTATIONS):
            theta = theta_idx * np.pi / GABOR_ORIENTATIONS
            sigma = size / 3.0
            lambd = size / 1.5
            gamma = 0.5

            half = size // 2
            f = np.zeros((size, size))
            for y in range(-half, half + 1):
                for x in range(-half, half + 1):
                    x_theta = x * np.cos(theta) + y * np.sin(theta)
                    y_theta = -x * np.sin(theta) + y * np.cos(theta)
                    f[y + half, x + half] = np.exp(
                        -(x_theta**2 + gamma**2 * y_theta**2) / (2 * sigma**2)
                    ) * np.cos(2 * np.pi * x_theta / lambd)
            f -= f.mean()
            norm = np.linalg.norm(f)
            if norm > 0:
                f /= norm
            filters.append(f)
    return filters


@njit(cache=True)
def _apply_filter(image, filt, stride):
    ih, iw = image.shape
    fh, fw = filt.shape
    oh = (ih - fh) // stride + 1
    ow = (iw - fw) // stride + 1
    output = np.zeros((oh, ow))
    for y in range(oh):
        for x in range(ow):
            val = 0.0
            for fy in range(fh):
                for fx in range(fw):
                    val += image[y * stride + fy, x * stride + fx] * filt[fy, fx]
            output[y, x] = max(0.0, val)
    return output


@njit(cache=True)
def _max_pool_2d(feature_map, pool_size):
    h, w = feature_map.shape
    oh = h // pool_size
    ow = w // pool_size
    output = np.zeros((oh, ow))
    for y in range(oh):
        for x in range(ow):
            max_val = -1e10
            for py in range(pool_size):
                for px in range(pool_size):
                    val = feature_map[y * pool_size + py, x * pool_size + px]
                    if val > max_val:
                        max_val = val
                    output[y, x] = max_val
    return output


def extract_features(image_flat, img_size, filters, stride=2, pool_size=0):
    image = image_flat.reshape(img_size, img_size)
    features = []
    for f in filters:
        response = _apply_filter(image, f, stride)
        if pool_size > 1 and response.shape[0] >= pool_size:
            response = _max_pool_2d(response, pool_size)
        features.append(response.flatten())
    return np.concatenate(features)


class VisualCortex:
    def __init__(self, img_size=16, stride=2, evolved_filters=None):
        self.img_size = img_size
        self.stride = stride
        if evolved_filters is not None:
            self.filters = evolved_filters
        else:
            self.filters = make_gabor_bank()
        self.num_filters = len(self.filters)

        test_img = np.zeros((img_size, img_size))
        sample_out = extract_features(test_img.flatten(), img_size, self.filters, stride)
        self.output_size = len(sample_out)

    def process(self, image_flat):
        features = extract_features(image_flat, self.img_size, self.filters, self.stride)
        if features.max() > 0:
            features = features / features.max()
        return features

    def process_batch(self, images):
        results = np.zeros((len(images), self.output_size))
        for i in range(len(images)):
            results[i] = self.process(images[i])
        return results


class EvolvedVisualCortex(VisualCortex):
    def __init__(self, img_size=16, stride=2, num_filters=8, filter_size=3, seed=42):
        rng = np.random.default_rng(seed)
        filters = []
        for _ in range(num_filters):
            f = rng.normal(0, 0.5, (filter_size, filter_size))
            f -= f.mean()
            norm = np.linalg.norm(f)
            if norm > 0:
                f /= norm
            filters.append(f)
        super().__init__(img_size, stride, evolved_filters=filters)

    def get_filter_params(self):
        return [f.copy() for f in self.filters]

    def set_filter_params(self, filter_list):
        self.filters = [f.copy() for f in filter_list]
        self.num_filters = len(filter_list)
        test_img = np.zeros((self.img_size, self.img_size))
        sample_out = extract_features(test_img.flatten(), self.img_size, self.filters, self.stride)
        self.output_size = len(sample_out)

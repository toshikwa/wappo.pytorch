import numpy as np


def tile_images(nhwc):
    nhwc = np.asarray(nhwc)
    N, h, w, c = nhwc.shape
    H = int(np.ceil(np.sqrt(N)))
    W = int(np.ceil(float(N)/H))
    nhwc = np.array(list(nhwc) + [nhwc[0] * 0 for _ in range(N, H*W)])
    HWhwc = np.reshape(nhwc, (H, W, h, w, c))
    HhWwc = HWhwc.transpose(0, 2, 1, 3, 4)
    Hh_Ww_c = np.reshape(HhWwc, (H*h, W*w, c))
    return Hh_Ww_c


def tile_images_torch(nchw):
    nchw = np.asarray(nchw)
    N, c, h, w = nchw.shape
    H = int(np.ceil(np.sqrt(N)))
    W = int(np.ceil(float(N)/H))
    HW_chw = np.array(list(nchw) + [nchw[0] * 0 for _ in range(N, H*W)])
    cHWhw = np.reshape(HW_chw.transpose(1, 0, 2, 3), (c, H, W, h, w))
    cHhWw = cHWhw.transpose(0, 1, 3, 2, 4)
    c_Hh_Ww = np.reshape(cHhWw, (c, H*h, W*w))
    return c_Hh_Ww


def normalize(tensor):
    mean = tensor.mean()
    std = tensor.std()
    return tensor.add_(-mean).div_(std + 1.e-8)


class LRAnneaer:

    def __init__(self, optim, start_value, end_value, num_steps):
        assert num_steps > 0 and isinstance(num_steps, int)

        self.steps = 0
        self.optim = optim
        self.start_value = start_value
        self.end_value = end_value
        self.num_steps = num_steps

        self.a = (self.end_value - self.start_value) / self.num_steps
        self.b = self.start_value

    def step(self):
        self.steps = min(self.num_steps, self.steps + 1)
        for param_group in self.optim.param_groups:
            param_group['lr'] = self.get()

    def get(self):
        assert 0 < self.steps <= self.num_steps
        return self.a * self.steps + self.b

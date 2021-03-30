import numpy as np


def enhance(im, stdval=6.0, just_alpha=True):
    if just_alpha:
        nz = im[im > 0.0]
        im[:] = im[:] / (nz.mean() + stdval * np.std(nz))
    else:
        for c in range(3):
            nz = im[:, :, c][im[:, :, c] > 0.0]
            im[:, :, c] = im[:, :, c] / (nz.mean() + stdval * np.std(nz))
            del nz
    np.clip(im, 0.0, 1.0, im)


def enhance_rgba(im, stdval=6.0):
    nzc = im[:, :, :3][im[:, :, :3] > 0.0]
    cmax = nzc.mean() + stdval * nzc.std()

    nza = im[:, :, 3][im[:, :, 3] > 0.0]
    if len(nza) == 0:
        im[:, :, 3] = 1.0
        amax = 1.0
    else:
        amax = nza.mean() + stdval * nza.std()

    im.rescale(amax=amax, cmax=cmax, inline=True)
    np.clip(im, 0.0, 1.0, im)


class PostEffect:
    def __init__(self):
        pass

    def apply(self):
        raise NotImplementedError


class Attenuate(PostEffect):
    def __init__(self, Cfog=(0.0, 0.0, 0.0, 1.0), att_fac=0.5, method="exp"):
        super().__init__()

        assert len(Cfog) == 4
        self.Cfog = Cfog
        self.att_fac = att_fac
        self.method = method

    def calculate_factor(self, zbuffer_dist):
        # simple attenuation factors following fog effects from
        # https://www.opengl.org/archives/resources/code/samples/sig99/advanced99/notes/node334.html
        if self.method == "exp":
            f = np.exp(-(self.att_fac * zbuffer_dist))
        elif self.method == "exp2":
            f = np.exp(-((self.att_fac * zbuffer_dist) ** 2))
        elif self.method == "linear":
            zmin = zbuffer_dist.min()
            zmax = zbuffer_dist[zbuffer_dist != np.inf].max()
            f = 1 - (zbuffer_dist - zmin) / (zmax - zmin)
            f[f == np.inf] = 1.0
            f[f == -np.inf] = 1.0
        return f

    def apply(self, zbuffer):
        f = self.calculate_factor(zbuffer.z)
        for c_ax in range(0, 3):
            zbuffer.rgba[:, :, c_ax] = (
                f * zbuffer.rgba[:, :, c_ax] + (1 - f) * self.Cfog[c_ax]
            )
        return zbuffer

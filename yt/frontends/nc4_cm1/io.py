from yt.frontends.nc4_cf.io import NCCFIOHandler


class CM1IOHandler(NCCFIOHandler):
    _dataset_type = "nc4_cm1"

    def __init__(self, ds):
        super().__init__(ds)

from yt.frontends.nc4_cf.io import NCCFIOHandler


class CM1IOHandler(NCCFIOHandler):
    # not doing anything different (yet....)
    def __init__(self, ds):
        super().__init__(ds)

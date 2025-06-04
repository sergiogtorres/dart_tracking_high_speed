import numpy as np

def histogram_equalization(ims):
    H, bins = np.histogram(ims.flatten(), range=[0,256], bins=256)

    cdf = (255 * np.cumsum(H/H.sum())).astype("uint8")

    ims_corrected = cdf[ims]

    return ims_corrected


def histogram_equalization_simple(ims, extra=1):
    bot = ims.min()
    top = ims.max()
    ims_corrected = extra*(ims-bot).astype(float)*255/(top-bot)

    return ims_corrected.astype("uint8")
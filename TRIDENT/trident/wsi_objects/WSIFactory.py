
import os
from typing import Optional, Literal, Union

from trident.wsi_objects.OpenSlideWSI import OpenSlideWSI
from trident.wsi_objects.ImageWSI import ImageWSI
from trident.wsi_objects.CuCIMWSI import CuCIMWSI
from trident.wsi_objects.SDPCWSI import SDPCWSI

WSIReaderType = Literal['openslide', 'image', 'cucim', 'sdpc']
OPENSLIDE_EXTENSIONS = {'.svs', '.tif', '.tiff', '.ndpi', '.vms', '.vmu', '.scn', '.mrxs'}
CUCIM_EXTENSIONS = {'.svs', '.tif', '.tiff'}
SDPC_EXTENSIONS = {'.sdpc'}
PIL_EXTENSIONS = {'.png', '.jpg', '.jpeg'}


def load_wsi(
    slide_path: str,
    reader_type: Optional[WSIReaderType] = None,
    **kwargs
) -> Union[OpenSlideWSI, ImageWSI, CuCIMWSI, SDPCWSI]:
    """
    Load a whole-slide image (WSI) using the appropriate backend.

    By default, uses OpenSlideWSI for OpenSlide-supported file extensions,
    and ImageWSI for others. Users may override this behavior by explicitly
    specifying a reader using the `reader_type` argument.

    Parameters
    ----------
    slide_path : str
        Path to the whole-slide image.
    reader_type : {'openslide', 'image', 'cucim', 'sdpc'}, optional
        Manually specify the WSI reader to use. If None (default), selection
        is automatic based on file extension.
    **kwargs : dict
        Additional keyword arguments passed to the WSI reader constructor.

    Returns
    -------
    Union[OpenSlideWSI, ImageWSI, CuCIMWSI, SDPCWSI]
        An instance of the appropriate WSI reader.

    Raises
    ------
    ValueError
        If `reader_type` is 'cucim' but the cucim package is not installed.
        Or if `reader_type` is 'sdpc' but the sdpc package is not installed.
        Or if an unknown reader type is specified.
    """
    ext = os.path.splitext(slide_path)[1].lower()

    assert reader_type in ['openslide', 'image', 'cucim', 'sdpc', None], f"Unknown reader_type: {reader_type}. Choose from 'openslide', 'image', 'cucim', or 'sdpc'."

    if reader_type == 'openslide':
        return OpenSlideWSI(slide_path=slide_path, **kwargs)

    elif reader_type == 'image':
        return ImageWSI(slide_path=slide_path, **kwargs)
    
    elif reader_type == 'sdpc':
        if ext in SDPC_EXTENSIONS:
            return SDPCWSI(slide_path=slide_path, **kwargs)
        else:
            raise ValueError(
                f"Unsupported file format '{ext}' for SDPC. "
                f"Supported whole-slide image formats are: {', '.join(SDPC_EXTENSIONS)}."
            )

    elif reader_type == 'cucim':
        if ext in CUCIM_EXTENSIONS:
            return CuCIMWSI(slide_path=slide_path, **kwargs)
        else:
            raise ValueError(
                f"Unsupported file format '{ext}' for CuCIM. "
                f"Supported whole-slide image formats are: {', '.join(CUCIM_EXTENSIONS)}."
            )
 
    elif reader_type is None:
        if ext in OPENSLIDE_EXTENSIONS:
            return OpenSlideWSI(slide_path=slide_path, **kwargs)
        elif ext in SDPC_EXTENSIONS:
            return SDPCWSI(slide_path=slide_path, **kwargs)
        else:
            return ImageWSI(slide_path=slide_path, **kwargs)

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
from shapely.geometry import LineString, Polygon, MultiPolygon


def prepare_image(
    image: Image, size: int = 300, shades: int = 16, flip: bool = True
) -> Image:
    """resizes, converts to grayscale and quantizes the image to a given number of shades. Optionally flips the image.

    Args:
        image (_type_): Image to prepare
        size (int, optional): Size to achieve. Defaults to 300.
        shades (int, optional): Number of shades of grey. Defaults to 16.
        flip (bool, optional): Boolean: should it flip? Defaults to True.
    """
    image = image.resize((size, size), Image.LANCZOS)
    image = image.convert("L")
    image = image.quantize(shades)
    if flip:
        image = ImageOps.flip(image)
    return image


def polygony(
    image: Image, size: int = 300, shades: int = 16, flip: bool = True
) -> list:
    """Converts an image to a list of polygons.

    Args:
        image (_type_): Image to convert
        size (int, optional): Size to achieve. Defaults to 300.
        shades (int, optional): Number of shades of grey. Defaults to 16.
        flip (bool, optional): Boolean: should it flip? Defaults to True.

    Returns:
        list: List of polygons
    """
    image = prepare_image(image, size, shades, flip)
    image = np.array(image).astype(np.float32) / 255.0
    polygons = []
    for y in range(size):
        for x in range(size):
            if image[y, x] > 0:
                polygons.append(
                    Polygon(
                        [
                            (x / size, y / size),
                            ((x + 1) / size, y / size),
                            ((x + 1) / size, (y + 1) / size),
                            (x / size, (y + 1) / size),
                        ]
                    )
                )
    return polygons


if __name__ == "__main__":
    image = Image.open("test.png")
    image = prepare_image(image)
    polygons = polygony(image)
    gdf = gpd.GeoDataFrame(geometry=polygons)
    gdf.plot()
    plt.show()

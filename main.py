import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
import rasterio
from shapely.geometry import LineString, Polygon, MultiPolygon, box
from shapely.ops import polygonize



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

def raster_to_geodataframe(raster_array):
    polygons = []
    for y in range(raster_array.shape[0]):
        for x in range(raster_array.shape[1]):
            if raster_array[y, x] > 0:
                polygons.append(
                    {
                        "geometry": box(x, y, x + 1, y + 1),
                        "col": raster_array[y, x],
                    }
                )
    return gpd.GeoDataFrame(polygons)


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
    # save image as test2.png
    image.save("test2.png")
    with rasterio.open("test2.png") as src:
        red_band = src.read(1)
        height, width = red_band.shape
        rescaled_red_band = 1 - (red_band - red_band.min()) / (red_band.max() - red_band.min())
    i_sf = raster_to_geodataframe(rescaled_red_band)
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

def plot_polygons(polygons):
    fig, ax = plt.subplots()
    polygons.plot(column="col", cmap="viridis_r", ax=ax, edgecolor="none")
    plt.show()

if __name__ == "__main__":
    image = Image.open("test.png")
    # Print image
    plt.imshow(image)
    plt.show()
    image = prepare_image(image, flip=True)
    # Print prepared image
    plt.imshow(image)
    plt.show()
    polygons = polygony(image)
    # Print image obtained from the polygon
    plot_polygons(polygons)


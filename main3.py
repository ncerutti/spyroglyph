import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import rasterio
import rasterio.features
import geopandas as gpd
from shapely.geometry import LineString, point, shape
from shapely.ops import cascaded_union, unary_union


def prepare_image(img, size, shades, invert):
    i = img.resize((size, size), Image.LANCZOS)
    i = i.quantize(shades)
    i = i.convert("L")
    if invert:
        i = Image.fromarray(255 - np.array(i))
    return i


def raster_to_geodataframe(image, rescaled_red_band):
    mask = rescaled_red_band > 0
    shapes = rasterio.features.shapes(
        rescaled_red_band.astype(np.float32), mask=mask, transform=image.transform
    )
    polygons = [{"geometry": shape(s), "col": v} for s, v in shapes]
    return gpd.GeoDataFrame(polygons)


def polygony(
    image: Image, size: int = 300, shades: int = 16, flip: bool = True
) -> list:
    image = prepare_image(image, size, shades, flip)
    # Flip image top to bottom
    image = image.transpose(Image.FLIP_TOP_BOTTOM)
    image.save("test2.png")

    with rasterio.open("test2.png") as src:
        red_band = src.read(1)
        # rescaled_red_band = 1 - (red_band - red_band.min()) / (
        #     red_band.max() - red_band.min()
        # )
        rescaled_red_band = (red_band - red_band.min()) / (
            red_band.max() - red_band.min()
        )

    i_sf = raster_to_geodataframe(src, rescaled_red_band)
    return i_sf


def plot_spiral_and_polygons(spiral_coords, polygons_gdf):
    # Convert the spiral coordinates to a LineString
    spiral = LineString(spiral_coords.values)

    # Create a GeoDataFrame with the spiral
    gdf_spiral = gpd.GeoDataFrame({"geometry": [spiral]})

    # Plot the polygons and spiral
    fig, ax = plt.subplots()
    polygons_gdf.plot(
        ax=ax, column="col", cmap="viridis_r", alpha=0.75, edgecolor="none"
    )
    gdf_spiral.plot(ax=ax, color="black", linewidth=1)

    # Customize the plot appearance
    ax.set_axis_off()
    plt.tight_layout()
    plt.show()


def plot_polygons(polygons):
    fig, ax = plt.subplots()
    polygons.plot(column="col", cmap="viridis_r", ax=ax, edgecolor="none")
    plt.show()


def spiral_coords(xo, yo, n_points, n_turns, r0, r1, offset_angle, scale=1):
    b = (r1 - r0) / (2 * np.pi * n_turns)
    l = np.linspace(0, 2 * np.pi * n_turns, num=n_points)

    x = (r0 + (b * l)) * np.cos(l + np.radians(offset_angle)) * scale + xo
    y = (r0 + (b * l)) * np.sin(l + np.radians(offset_angle)) * scale + yo

    return pd.DataFrame({"x": x, "y": y})


def coords_to_gdf_spiral(coords):
    # Convert the coordinates to a LineString
    spiral_linestring = LineString(coords)

    # Create a GeoDataFrame from the LineString
    gdf_spiral = gpd.GeoDataFrame(geometry=[spiral_linestring], crs="EPSG:4326")

    return gdf_spiral


def buffered_intersections(polygons_gdf, gdf_spiral, n_turns):
    intersections = gpd.overlay(
        polygons_gdf, gdf_spiral, how="intersection", keep_geom_type=False
    )

    if not intersections.empty:
        thin = 0.00025 * scale_factor
        thick = ((0.5 / n_turns) / 2) * 0.50 * scale_factor

        intersections["n"] = intersections["col"].apply(
            lambda x: (thick - thin) * x + thin
        )
        intersections["geometry"] = intersections.geometry.buffer(
            intersections["n"], cap_style=2
        )

        return intersections
    else:
        return None


# Test the function
if __name__ == "__main__":
    img = Image.open("test.png")
    imaprep = prepare_image(img, 300, 16, True)
    plt.imshow(imaprep)
    plt.show()
    polygons_gdf = polygony(img)

    bounds = polygons_gdf.total_bounds
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    center_x = (bounds[0] + bounds[2]) / 2
    center_y = (bounds[1] + bounds[3]) / 2

    # Adjust the scale factor if needed
    scale_factor = max(width, height)
    coords = spiral_coords(center_x, center_y, 5000, 33, 0, 0.5, 0, scale=scale_factor)
    print(polygons_gdf)
    plot_polygons(polygons_gdf)

    # Extract the list of geometries from the GeoDataFrame
    polygons = polygons_gdf["geometry"].tolist()
    plot_spiral_and_polygons(coords, polygons_gdf)

    gdf_spiral = coords_to_gdf_spiral(coords)
    plot_polygons(polygons_gdf)
    intersections = buffered_intersections(polygons_gdf, gdf_spiral, 50)

    # Plot the buffered intersections
    fig, ax = plt.subplots()
    intersections.plot(ax=ax, facecolor="black", edgecolor="none", cmap="viridis_r")
    ax.set_aspect("equal")
    ax.set_axis_off()
    plt.tight_layout()
    plt.show()

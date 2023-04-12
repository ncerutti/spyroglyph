import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import geopandas as gpd
from shapely.geometry import LineString, Polygon, MultiPolygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union


def spiral_image(
    img,
    invert=False,
    size=300,
    n_shades=16,
    spiral_points=5000,
    spiral_turns=50,
    spiral_r0=0,
    spiral_r1_f=1,
    thin=0.00025,
    thick_f=0.95,
    spiral_offset_angle=0,
    col_line="black",
    col_bg="white",
):
    # Read image
    if isinstance(img, Image.Image):
        i = img
    else:
        i = Image.open(img)

    # Process image to Shapely polygon
    i = i.resize((size, size), Image.LANCZOS)
    i = i.convert("L")
    i = i.quantize(n_shades)
    i = np.array(i).astype(np.float32)

    if invert:
        i = 1 - i / 255
    else:
        i /= 255

    polygons = []
    for y in range(size):
        for x in range(size):
            if i[y, x] > 0:
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

    # Generate spiral
    def spiral_coords(xo, yo, n_points, n_turns, r0, r1, offset_angle):
        theta = np.linspace(0, 2 * np.pi * n_turns, n_points)
        r = np.linspace(r0, r1, n_points)
        x = xo + r * np.cos(theta + offset_angle)
        y = yo + r * np.sin(theta + offset_angle)
        return np.column_stack((x, y))

    spiral = spiral_coords(
        0.5,
        0.5,
        spiral_points,
        spiral_turns,
        spiral_r0,
        0.5 * spiral_r1_f,
        spiral_offset_angle,
    )
    spiral = LineString(spiral)

    # Intersect polygons with spiral
    gdf = gpd.GeoDataFrame({"geometry": polygons})
    intersections = gdf[gdf.intersects(spiral)].copy()
    intersections["n"] = intersections.apply(
        lambda row: row.geometry.intersection(spiral).length, axis=1
    )
    intersections["n"] = intersections["n"].apply(lambda x: x * (thick_f - thin) + thin)

    buffered_polygons = []
    for _, row in intersections.iterrows():
        buffered_intersection = row.geometry.intersection(spiral).buffer(
            row["n"], cap_style=1
        )
        if isinstance(buffered_intersection, Polygon):
            buffered_polygons.extend([buffered_intersection])
        elif isinstance(buffered_intersection, MultiPolygon):
            buffered_polygons.extend(
                [
                    geom
                    for geom in buffered_intersection.geoms
                    if isinstance(geom, BaseGeometry)
                ]
            )

    buffered_intersections = unary_union(buffered_polygons)

    # for _, row in intersections.iterrows():
    #     buffered_line = row.geometry.intersection(spiral).buffer(row["n"], cap_style=1)
    #     buffered_polygons.extend([geom for geom in buffered_line if isinstance(geom, BaseGeometry)])

    # buffered_intersections = unary_union(buffered_polygons)

    # Plot the image
    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    ax.set_facecolor(col_bg)

    if isinstance(buffered_intersections, Polygon):
        geom_list = [buffered_intersections]
    elif isinstance(buffered_intersections, list):
        geom_list = buffered_intersections
    else:
        geom_list = list(buffered_intersections.geoms)

    for polygon in geom_list:
        x, y = polygon.exterior.xy
        ax.fill(x, y, col_line)
    plt.show()


if __name__ == "__main__":
    spiral_image("test.png", invert=True, col_bg="white", col_line="black")

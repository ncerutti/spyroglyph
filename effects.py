""" Here are the effects that can be applied to the images"""
import io
import matplotlib.pyplot as plt
import noise
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import rasterio
import rasterio.features
import geopandas as gpd
from shapely.geometry import LineString, Point, Polygon, shape


def prepare_image(img, size, shades, crop=False):
    """_summary_

    Args:
        img (PIL image): input image
        size (int): target size
        shades (int): number of shades
        crop (bool, optional): Should the image be cropped instead of resized? Defaults to False.

    Returns:
        i (PIL image): black and white, quantized, resized image
    """
    if crop:
        i = img.crop((0, 0, size, size))
    else:
        i = img.resize((size, size))
    i = i.quantize(shades)
    i = i.convert("L")
    return i


def raster_to_geodataframe(image, rescaled_red_band):
    """Use rasterio to convert a raster image to a GeoDataFrame

    Args:
        image (_type_): PIL image
        rescaled_red_band (_type_): one of the channels

    Returns:
        GeoDataFrame object
    """
    mask = rescaled_red_band > 0
    shapes = rasterio.features.shapes(
        rescaled_red_band.astype(np.float32), mask=mask, transform=image.transform
    )
    polygons = [{"geometry": shape(s), "col": v} for s, v in shapes]
    return gpd.GeoDataFrame(polygons)


def polygony(image: Image, rescaler_factor=1.0) -> list:
    """Convert the input PIL Image into a GeoDataFrame of polygons.

    Args:
        image (PIL Image): Input image
    rescaler_factor (float, optional): Rescaling factor for rescaled_red_band. Defaults to 1.0.

    Returns:
        list: List of polygons as a GeoDataFrame object
    """
    image = image.transpose(Image.FLIP_TOP_BOTTOM)
    image.save("test2.png")
    with rasterio.open("test2.png") as src:
        red_band = src.read(1)
        # rescaled_red_band = 1 - (red_band - red_band.min()) / (
        #     red_band.max() - red_band.min()
        # )
        rescaled_red_band = rescaler_factor - (red_band - red_band.min()) / (
            red_band.max() - red_band.min()
        )
    i_sf = raster_to_geodataframe(src, rescaled_red_band)
    return i_sf


def plot_spiral_and_polygons(spiral_coords, polygons_gdf):
    """Plot the given spiral coordinates and polygons GeoDataFrame. Mostly for debugging.

    Args:
    spiral_coords (pd.DataFrame): DataFrame containing the spiral coordinates
    polygons_gdf (geopandas.GeoDataFrame): GeoDataFrame containing polygons

    """

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
    """Plot the given polygons GeoDataFrame. Mostly for debugging.

    Args:
        polygons (geopandas.GeoDataFrame): GeoDataFrame containing polygons
    """
    fig, ax = plt.subplots()
    polygons.plot(column="col", cmap="viridis_r", ax=ax, edgecolor="none")
    plt.show()


def spiral_coords(xo, yo, n_points, n_turns, r0, r1, offset_angle, scale=1):
    """Generate the coordinates for a spiral.

    Args:
        xo (float): X-coordinate of the spiral's center
        yo (float): Y-coordinate of the spiral's center
        n_points (int): Number of points in the spiral
        n_turns (int): Number of turns in the spiral
        r0 (float): Initial radius of the spiral
        r1 (float): Final radius of the spiral
        offset_angle (float): Angle to offset the spiral, in degrees
        scale (float, optional): Scaling factor for the spiral. Defaults to 1.

    Returns:
        pd.DataFrame: DataFrame containing the coordinates of the spiral
    """
    b = (r1 - r0) / (2 * np.pi * n_turns)
    l = np.linspace(0, 2 * np.pi * n_turns, num=n_points)

    x = (r0 + (b * l)) * np.cos(l + np.radians(offset_angle)) * scale + xo
    y = (r0 + (b * l)) * np.sin(l + np.radians(offset_angle)) * scale + yo

    return pd.DataFrame({"x": x, "y": y})


def coords_to_gdf_spiral(coords):
    """Converts the given coordinates into a GeoDataFrame containing a LineString.

    Args:
        xo (float): X-coordinate of the spiral's center
        yo (float): Y-coordinate of the spiral's center
        n_points (int): Number of points in the spiral
        n_turns (int): Number of turns in the spiral
        r0 (float): Initial radius of the spiral
        r1 (float): Final radius of the spiral
        offset_angle (float): Angle to offset the spiral, in degrees
        scale (float, optional): Scaling factor for the spiral. Defaults to 1.

    Returns:
        pd.DataFrame: DataFrame containing the coordinates of the spiral
    """

    # Convert the coordinates to a LineString
    spiral_linestring = LineString(coords)

    # Create a GeoDataFrame from the LineString
    gdf_spiral = gpd.GeoDataFrame(geometry=[spiral_linestring], crs="EPSG:4326")

    return gdf_spiral


def buffered_intersections(
    polygons_gdf, gdf_spiral, n_turns, scale_factor, thin, thick, spiral_r1
):
    """Calculate buffered intersections between polygons and spiral.

    Args:
        polygons_gdf (geopandas.GeoDataFrame): GeoDataFrame containing polygons
        gdf_spiral (geopandas.GeoDataFrame): GeoDataFrame containing the spiral
        n_turns (int): Number of turns in the spiral
        scale_factor (float): Scaling factor for the spiral
        thin (float): Minimum buffer width
        thick (float): Maximum buffer width
        spiral_r1 (float): Final radius of the spiral

    Returns:
        geopandas.GeoDataFrame: Buffered intersections"""
    intersections = gpd.overlay(
        polygons_gdf, gdf_spiral, how="intersection", keep_geom_type=False
    )

    if not intersections.empty:
        thin = thin
        thick = ((spiral_r1 / n_turns) / 2) * thick * scale_factor

        intersections["n"] = intersections["col"].apply(
            lambda x: (thick - thin) * x + thin
        )
        intersections["geometry"] = intersections.geometry.buffer(
            intersections["n"], cap_style=2
        )

        return intersections
    else:
        return None


def spiral_function(
    input_image="test.png",
    size=300,
    n_shades=16,
    spiral_points=5000,
    spiral_turns=50,
    spiral_r0=0,
    spiral_r1_f=0.5,
    thin=0.00025,
    thick_f=0.95,
    spiral_offset_angle=0,
    crop=False,
    color="black",
    colormap="gray",
    output_image="output.png",
    rescaler_factor=1.0,
    alpha=0.75,
):
    """
    Args:
        image (_type_): _description_
        n_turns (int, optional): _description_. Defaults to 50.
        n_points (int, optional): _description_. Defaults to 5000.
        size (int, optional): _description_. Defaults to 300.
        invert (_type_, optional): _description_. Defaults to FALSE.
        size (int, optional): _description_. Defaults to 300.
        n_shades (int, optional): _description_. Defaults to 16.
        spiral_points (int, optional): _description_. Defaults to 5000.
        spiral_turns (int, optional): _description_. Defaults to 50.
        spiral_r0 (int, optional): _description_. Defaults to 0.
        spiral_r1_f (int, optional): _description_. Defaults to 1.
        thin (float, optional): _description_. Defaults to 0.00025.
        thick_f (float, optional): _description_. Defaults to 0.95.
        spiral_offset_angle (int, optional): _description_. Defaults to 0.
        crop (bool, optional): _description_. Defaults to False.
        colormap (str, optional): _description_. Defaults to "gray".
        output_image (str, optional): _description_. Defaults to "output.png".
        rescaler_factor (float, optional): _description_. Defaults to 1.0.
    """
    # Prepare the image
    img = Image.open(input_image)
    img = prepare_image(img, size=size, shades=n_shades, crop=crop)
    polygons_gdf = polygony(img, rescaler_factor=rescaler_factor)
    try:
        bounds = polygons_gdf.total_bounds
    except ValueError:
        print("ValueError: No polygons found.")
        return None
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    center_x = (bounds[0] + bounds[2]) / 2
    center_y = (bounds[1] + bounds[3]) / 2
    scale_factor = max(width, height)
    coords = spiral_coords(
        center_x,
        center_y,
        spiral_points,
        spiral_turns,
        spiral_r0,
        spiral_r1_f,
        spiral_offset_angle,
        scale=scale_factor,
    )
    gdf_spiral = coords_to_gdf_spiral(coords)
    intersections = buffered_intersections(
        polygons_gdf,
        gdf_spiral,
        spiral_turns,
        scale_factor,
        thin,
        thick_f,
        spiral_r1=spiral_r1_f,
    )
    fig, ax = plt.subplots()
    if colormap == "none":
        intersections.plot(ax=ax, facecolor=color, edgecolor="none", alpha=alpha)
    else:
        intersections.plot(
            ax=ax, facecolor=color, edgecolor="none", cmap=colormap, alpha=alpha
        )
    ax.set_aspect("equal")
    ax.set_axis_off()
    plt.tight_layout()
    plt.show()
    fig.savefig(output_image, dpi=300, bbox_inches="tight", pad_inches=0)


def double_spiral_function(
    input_image_1="test_a.png",
    input_image_2="test_b.png",
    size=300,
    n_shades=16,
    spiral_points=5000,
    spiral_turns=50,
    spiral_r0=0,
    spiral_r1_f=0.5,
    thin=0.00025,
    thick_f=0.5,
    spiral_offset_angle=0,
    crop=False,
    color_1="gray",
    color_2="gray",
    alpha_1=0.75,
    alpha_2=0.5,
    output_image="output.png",
    rescaler_factor=1.0,
):
    # Prepare the image
    img_a = Image.open(input_image_1)
    img_a = prepare_image(img_a, size=size, shades=n_shades, crop=crop)
    polygons_gdf_a = polygony(img_a, rescaler_factor=rescaler_factor)
    img_b = Image.open(input_image_2)
    img_b = prepare_image(img_b, size=size, shades=n_shades, crop=crop)
    polygons_gdf_b = polygony(img_b, rescaler_factor=rescaler_factor)
    try:
        bounds = polygons_gdf_a.total_bounds
    except ValueError:
        print("ValueError: No polygons found.")
        return None
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    center_x = (bounds[0] + bounds[2]) / 2
    center_y = (bounds[1] + bounds[3]) / 2
    scale_factor = max(width, height)
    coords = spiral_coords(
        center_x,
        center_y,
        spiral_points,
        spiral_turns,
        spiral_r0,
        spiral_r1_f,
        spiral_offset_angle,
        scale=scale_factor,
    )
    gdf_spiral = coords_to_gdf_spiral(coords)
    intersections_positive = buffered_intersections(
        polygons_gdf_a,
        gdf_spiral,
        spiral_turns,
        scale_factor,
        thin,
        thick_f,
        spiral_r1=spiral_r1_f,
    )

    # Create intersections with positive and negative buffer values
    intersections_positive["n"] = intersections_positive["col"].apply(
        lambda x: (thick_f - thin) * x + thin
    )

    intersections_positive["geometry"] = intersections_positive.geometry.buffer(
        intersections_positive["n"], cap_style=2, single_sided=True
    )

    intersections_negative = buffered_intersections(
        polygons_gdf_b,
        gdf_spiral,
        spiral_turns,
        scale_factor,
        thin,
        thick_f,
        spiral_r1=spiral_r1_f,
    )

    # intersections_negative["geometry"] = intersections_negative.geometry.buffer(
    #     -intersections_negative["n"], cap_style=2, single_sided=True
    # )

    # Remove Points from the intersections_negative
    intersections_negative = intersections_negative[
        intersections_negative["geometry"].apply(lambda x: not isinstance(x, Point))
    ]

    intersections_positive = intersections_positive[intersections_positive.is_valid]
    intersections_negative = intersections_negative[intersections_negative.is_valid]

    # Plot intersections with different colors
    fig, ax = plt.subplots()
    intersections_positive.plot(
        ax=ax, facecolor=color_1, edgecolor="none", alpha=alpha_1
    )
    intersections_negative.plot(
        ax=ax, facecolor=color_2, edgecolor="none", alpha=alpha_2
    )
    ax.set_aspect("equal")
    ax.set_axis_off()
    plt.tight_layout()
    plt.show()
    fig.savefig(output_image, dpi=300, bbox_inches="tight", pad_inches=0)


def grid_function(
    input_image="test.png",
    size=300,
    n_shades=16,
    grid_size=10,
    thin=0.00025,
    thick_f=0.95,
    grid_angle=0,
    crop=False,
    colormap="gray",
    output_image="output.png",
    rescaler_factor=1.0,
):
    """TBD

    Args:
        input_image (str, optional): _description_. Defaults to "test.png".
        size (int, optional): _description_. Defaults to 300.
        n_shades (int, optional): _description_. Defaults to 16.
        grid_size (int, optional): _description_. Defaults to 10.
        thin (float, optional): _description_. Defaults to 0.00025.
        thick_f (float, optional): _description_. Defaults to 0.95.
        grid_angle (int, optional): _description_. Defaults to 0.
        crop (bool, optional): _description_. Defaults to False.
        colormap (str, optional): _description_. Defaults to "gray".
        output_image (str, optional): _description_. Defaults to "output.png".
        rescaler_factor (float, optional): _description_. Defaults to 1.0.
    """
    pass


def create_noise_matrix(x_side, y_side):
    """
    Create a noise matrix using Perlin noise.

    Args:
        x_side (int): Width of the noise matrix.
        y_side (int): Height of the noise matrix.

    Returns:
        noise_matrix (numpy array): 2D noise matrix with the given dimensions.
    """
    noise_matrix = np.zeros((y_side, x_side))

    for i in range(y_side):
        for j in range(x_side):
            noise_matrix[i][j] = noise.snoise2(
                i * 0.0003, j * 0.0003, octaves=1, persistence=0.5, lacunarity=2.0
            )

    noise_matrix = np.interp(
        noise_matrix, (noise_matrix.min(), noise_matrix.max()), (-90, 90)
    )
    return noise_matrix


def flow_polygons(x_start, y_start, step_length, n_steps, angle_matrix):
    out_x = [x_start] + [np.nan] * n_steps
    out_y = [y_start] + [np.nan] * n_steps

    if (
        x_start > angle_matrix.shape[1]
        or x_start < 1
        or y_start > angle_matrix.shape[0]
        or y_start < 1
    ):
        return None

    for i in range(n_steps):
        a = angle_matrix[int(round(out_y[i])) - 1, int(round(out_x[i])) - 1]
        step_x = np.cos(np.radians(a)) * step_length
        step_y = np.sin(np.radians(a)) * step_length

        next_x = out_x[i] + step_x
        next_y = out_y[i] + step_y

        if (
            next_x > angle_matrix.shape[1]
            or next_x < 1
            or next_y > angle_matrix.shape[0]
            or next_y < 1
        ):
            break

        out_x[i + 1] = next_x
        out_y[i + 1] = next_y

    coords = np.column_stack((out_x, out_y))
    coords = coords[~np.isnan(coords).any(axis=1)]

    return coords


def flow_function(
    input_image,
    size=300,
    x_side=300,
    y_side=300,
    n_points=800,
    step_length=1,
    n_steps=400,
    n_shades=16,
    thin=0.0001,
    thick=0.0025,
    output_image="output_flow.png",
    crop=False,
    rescaler_factor=1.0,
    color = "black",
    alpha = 1.0,
    colormap="none",
):
    # Prepare the image
    img = Image.open(input_image)
    img = prepare_image(img, size=size, shades=n_shades, crop=crop)
    polygons_gdf = polygony(img, rescaler_factor=rescaler_factor)

    noise_matrix = create_noise_matrix(x_side, y_side)

    x_starts = np.random.uniform(1, noise_matrix.shape[1], n_points)
    y_starts = np.random.uniform(1, noise_matrix.shape[0], n_points)

    flow_lines = []

    for x_start, y_start in zip(x_starts, y_starts):
        coords = flow_polygons(x_start, y_start, step_length, n_steps, noise_matrix)
        if coords is not None and len(coords) > 1:
            line = LineString(coords)
            flow_lines.append(line)

    flow_gdf = gpd.GeoDataFrame(geometry=flow_lines)

    # Intersect flow lines with polygons
    intersections = gpd.overlay(
        polygons_gdf, flow_gdf, how="intersection", keep_geom_type=False
    )

    # Calculate line widths based on the 'col' value
    intersections["n"] = intersections["col"].apply(lambda x: (thick - thin) * x + thin)
    intersections["geometry"] = intersections.geometry.buffer(
        intersections["n"], cap_style=2
    )

    # Plot the intersections
    fig, ax = plt.subplots()
    if colormap == "none":
        intersections.plot(ax=ax, facecolor=color, edgecolor="none", alpha=alpha)
    else:
        intersections.plot(ax=ax, facecolor=color, edgecolor="none", cmap=colormap, alpha=alpha)
    ax.set_aspect("equal")
    ax.set_axis_off()
    plt.tight_layout()
    plt.show()
    fig.savefig(output_image, dpi=300, bbox_inches="tight", pad_inches=0)

    # Convert flow lines to a PIL image
    img_width, img_height = img.size
    flowed_image = Image.new("L", (img_width, img_height), 255)
    draw = ImageDraw.Draw(flowed_image)

    for line in flow_gdf.geometry:
        draw.line(line.coords, fill=0)

    # Convert the plotted figure to a PIL image
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight", pad_inches=0)
    buf.seek(0)
    flowed_image = Image.open(buf)

    return flowed_image

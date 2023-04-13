import streamlit as st
import tempfile
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import io
import rasterio
import rasterio.features
import geopandas as gpd
from shapely.geometry import LineString, Point, Polygon, shape

# from shapely.ops import cascaded_union, triangulate


def prepare_image(img, size, shades, crop=False):
    if crop:
        i = img.crop((0, 0, size, size))
    else:
        i = img.resize((size, size))
    i = i.quantize(shades)
    i = i.convert("L")
    return i


def raster_to_geodataframe(image, rescaled_red_band):
    mask = rescaled_red_band > 0
    shapes = rasterio.features.shapes(
        rescaled_red_band.astype(np.float32), mask=mask, transform=image.transform
    )
    polygons = [{"geometry": shape(s), "col": v} for s, v in shapes]
    return gpd.GeoDataFrame(polygons)


def polygony(image: Image, rescaler_factor=1.0) -> list:
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


def buffered_intersections(
    polygons_gdf, gdf_spiral, n_turns, scale_factor, thin, thick, spiral_r1
):
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


def spyroglyph(
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
    colormap="gray",
    output_image="output.png",
    rescaler_factor=1.0,
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
    # intersections.plot(ax=ax, facecolor="black", edgecolor="none", cmap="gray")
    intersections.plot(ax=ax, facecolor="black", edgecolor="none", cmap="gray")
    # intersections.plot(ax=ax, facecolor="black", edgecolor="none", cmap="plasma")
    ax.set_aspect("equal")
    ax.set_axis_off()
    plt.tight_layout()
    plt.show()
    fig.savefig(output_image, dpi=300, bbox_inches="tight", pad_inches=0)


def double_spyroglyph(
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
    colormap="gray",
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
    intersections_positive.plot(ax=ax, facecolor="blue", edgecolor="none", cmap="gray")
    intersections_negative.plot(ax=ax, facecolor="red", edgecolor="none")
    ax.set_aspect("equal")
    ax.set_axis_off()
    plt.tight_layout()
    plt.show()
    fig.savefig(output_image, dpi=300, bbox_inches="tight", pad_inches=0)


def main():
    st.title("Spyroglyph")

    uploaded_file = st.file_uploader("Choose a png image file", type=["png"])
    if uploaded_file is not None:
        input_image = Image.open(uploaded_file)
        st.image(input_image, caption="Uploaded Image", use_column_width=True)
        size = st.sidebar.slider("Size", 100, 500, 300)
        shades = st.sidebar.slider("Shades", 1, 64, 16)
        spiral_points = st.sidebar.slider("Spiral Points", 1000, 10000, 5000)
        spiral_turns = st.sidebar.slider("Spiral Turns", 10, 100, 50)
        spiral_r0 = st.sidebar.slider("Spiral r0", 0, 100, 0)
        spiral_r1_f = st.sidebar.slider("Spiral r1 factor", 0.0, 1.0, 0.5)
        thin = st.sidebar.slider("Thin", 0.0001, 0.0010, 0.00025)
        thick_f = st.sidebar.slider("Thick factor", 0.0, 1.0, 0.95)
        spiral_offset_angle = st.sidebar.slider("Spiral Offset Angle", 0, 360, 0)
        crop = st.sidebar.checkbox("Crop Image")
        colormap = st.sidebar.selectbox("Colormap", ["gray", "viridis", "plasma"])
        rescaler_factor = st.sidebar.slider("Rescaler Factor", 0.0, 2.0, 1.0)
        # Create a temporary file for the input image
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            input_image.save(temp_file.name)
            temp_file_path = temp_file.name

        if st.button("Generate Spyroglyph"):
            output_buffer = io.BytesIO()
            spyroglyph(
                input_image=temp_file_path,
                size=size,
                n_shades=shades,
                spiral_points=spiral_points,
                spiral_turns=spiral_turns,
                spiral_r0=spiral_r0,
                spiral_r1_f=spiral_r1_f,
                thin=thin,
                thick_f=thick_f,
                spiral_offset_angle=spiral_offset_angle,
                crop=crop,
                colormap=colormap,
                output_image=output_buffer,
                rescaler_factor=rescaler_factor,
            )
            output_image = Image.open(output_buffer)
            st.image(
                output_image, caption="Generated Spyroglyph", use_column_width=True
            )
            output_buffer.seek(0)
            st.download_button(
                "Download Spyroglyph", output_buffer, file_name="spyroglyph.png"
            )


if __name__ == "__main__":
    main()

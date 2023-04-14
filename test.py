from PIL import Image, ImageDraw
from effects import flow_function, plot_polygons
import matplotlib.pyplot as plt

flowed = flow_function(
    "testh.jpg",
    size=400,
    x_side=500,
    y_side=600,
    n_points=700,
    step_length=1,
    n_steps=1000,
    n_shades=16,
    thin=0.0025,
    thick=0.95,
    crop=False,
    rescaler_factor=1.0,
    output_image="output_flow_test.png",
)

plt.imshow(flowed)
plt.show()

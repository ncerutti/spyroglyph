from PIL import Image, ImageDraw
from effects import flow_function, plot_polygons, pixelate
import matplotlib.pyplot as plt

imagine = Image.open("testh.jpg")

flowed = pixelate(imagine, 20)

plt.imshow(flowed)
plt.show()

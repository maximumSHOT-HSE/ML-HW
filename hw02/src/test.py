from src.utils import *

if __name__ == '__main__':
    image = read_image("resources/image.jpg")
    result = clusterize_image(image)
    show_image(result)
    save_image(result, "resources/result.jpg")

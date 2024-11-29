import cv2
import numpy as np
from PIL import Image
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry


class Vectorizer:
    def __init__(self, model_path: str, model_type: str = "vit_l", max_size: int = 1024):
        self.model = sam_model_registry[model_type](checkpoint=model_path)
        self.mask_generator = SamAutomaticMaskGenerator(self.model)
        self.max_size = max_size
        self.image = None
        self.original_image = None

    def load_image(self, image_path: str):
        self.original_image = np.array(Image.open(image_path).convert('RGB'))
        h, w, _ = self.original_image.shape
        scale_factor = self.max_size / max(h, w)
        new_h = int(h * scale_factor)
        new_w = int(w * scale_factor)
        self.image = cv2.resize(self.original_image, (new_w, new_h))

    def calculate_average_color(self, mask: np.ndarray) -> tuple:
        mask_binary = mask.astype(bool)
        masked_pixels = self.image[mask_binary]
        if len(masked_pixels) > 0:
            average_color = np.mean(masked_pixels, axis=0).astype(int)
            return tuple(average_color)
        return (0, 0, 0)  # Default to black if no pixels are found

    def extract_contours(self, mask: np.ndarray):
        mask_uint8 = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def contour_to_svg_path(self, contour: np.ndarray) -> str:
        points = contour.reshape(-1, 2)
        path = f"M {points[0][0]},{points[0][1]}"
        path += " " + " ".join(f"L {x},{y}" for x, y in points[1:])
        path += " Z"
        return path

    def save_as_svg(self, masks: list, output_file: str):
        svg_width = self.image.shape[1]
        svg_height = self.image.shape[0]

        svg_parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {svg_width} {svg_height}" '
            f'width="{svg_width}" height="{svg_height}">'
        ]

        for mask_dict in masks:
            mask = mask_dict['segmentation']
            average_color = self.calculate_average_color(mask)
            color = f'rgb({average_color[0]},{average_color[1]},{average_color[2]})'

            contours = self.extract_contours(mask)
            for contour in contours:
                path = self.contour_to_svg_path(contour)
                svg_parts.append(f'<path d="{path}" fill="{color}" stroke="none" />')

        svg_parts.append('</svg>')

        with open(output_file, "w") as f:
            f.write('\n'.join(svg_parts))

    def generate_masks(self) -> list:
        return self.mask_generator.generate(self.image)

    def vectorize(self, image_path: str, output_file: str = "./output.svg"):
        self.load_image(image_path)
        masks = self.generate_masks()
        self.save_as_svg(masks, output_file)

# Example usage

if __name__ == "__main__":
    model_path = "./sam_vit_l_0b3195.pth"  # Update with your SAM model path
    vectorizer = Vectorizer(model_path)

    # Vectorize the uploaded image
    input_image_path = "/path/to/image.png"  # Replace with your image path
    output_svg_path = "./output.svg"  # Specify where to save the SVG

    vectorizer.vectorize(image_path=input_image_path, output_file=output_svg_path)

    print(f"SVG saved to {output_svg_path}")

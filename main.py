import numpy as np
import cv2
from PIL import ImageGrab
import time
import os

# Known RGB values for your squares (e.g., for covered cells, flagged cells, etc.)
RGB_BOARD1 = (162, 209, 73)  
RGB_BOARD2 = (170, 215, 81) 
RGB_BACK1 = (215, 184, 153)
RGB_BACK2 = (229, 194, 159)

# Define a tolerance for color matching 
TOLERANCE = 2

os.makedirs('covered_cells', exist_ok=True)
os.makedirs('flagged_cells', exist_ok=True)

def load_templates(template_path="templates"):
    templates = {}
    for i in range(9):
        template_file = os.path.join(template_path, f"{i}.png")
        if os.path.exists(template_file):
            template = cv2.imread(template_file, cv2.IMREAD_GRAYSCALE)
            templates[i] = template
        else:
            print(f"Template {i}.png not found.")
    return templates

def capture_screen(scale_percent=70):
    # Capture the screen (ImageGrab returns RGB format)
    screen = np.array(ImageGrab.grab())
    
    # Convert RGB to BGR for OpenCV compatibility (OpenCV uses BGR format)
    screen_bgr = cv2.cvtColor(screen, cv2.COLOR_RGB2BGR)
    
    # Resize the image to reduce size (default 50%)
    width = int(screen_bgr.shape[1] * scale_percent / 100)
    height = int(screen_bgr.shape[0] * scale_percent / 100)
    resized_screen = cv2.resize(screen_bgr, (width, height), interpolation=cv2.INTER_AREA)
    
    return resized_screen

def color_match(pixel, target_rgb, tolerance):
    # Check if a pixel matches the target RGB value within a tolerance
    r, g, b = pixel
    target_r, target_g, target_b = target_rgb
    return (abs(r - target_r) <= tolerance and
            abs(g - target_g) <= tolerance and
            abs(b - target_b) <= tolerance)

def segment_by_color(image, target_rgb, tolerance):
    # Create a binary mask where pixels of the target color are white and the rest are black
    mask = np.zeros(image.shape[:2], dtype=np.uint8)  # Mask to store results
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            pixel = image[y, x]
            if color_match(pixel, target_rgb, tolerance):
                mask[y, x] = 255  # Mark the pixel if it matches the target color
    return mask

def find_connected_components(mask):
    # Find connected components in the binary mask
    num_labels, labels = cv2.connectedComponents(mask)
    return num_labels, labels

# Function to extract and save bounding box regions as tiny images
def extract_bounding_box_image(image, x, y, w, h, label_type, idx, templates):
    # Crop the region of interest (ROI) from the original image
    roi = image[y:y+h, x:x+w]

    # Convert ROI to grayscale for template matching
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    best_match = None
    best_score = float('inf')

    # Perform template matching
    for number, template in templates.items():
        template = cv2.resize(template, (w, h), interpolation=cv2.INTER_AREA)
        res = cv2.matchTemplate(roi_gray, template, cv2.TM_SQDIFF_NORMED)
        min_val, _, _, _ = cv2.minMaxLoc(res)

        if min_val < best_score:
            best_score = min_val
            best_match = number

    if best_match is not None:
        # Construct the filename based on the match
        filename = f"{label_type}_component_{idx}_match_{best_match}.png"
    else:
        filename = f"{label_type}_component_{idx}_nomatch.png"

    filepath = os.path.join(label_type + '_cells', filename)
    cv2.imwrite(filepath, roi)
    print(f"Saved: {filepath}")


def detect_cells(image, templates):
    # Convert the image to RGB to match with the RGB values provided
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Segment the image by the RGB colors of the covered and flagged cells
    covered_mask = segment_by_color(image_rgb, RGB_BACK1, TOLERANCE)
    flagged_mask = segment_by_color(image_rgb, RGB_BACK2, TOLERANCE)

    # Find connected components for covered cells
    num_labels_covered, labels_covered = find_connected_components(covered_mask)

    # Find connected components for flagged cells
    num_labels_flagged, labels_flagged = find_connected_components(flagged_mask)

    # Draw bounding boxes for each connected component (covered and flagged cells)
    for i in range(1, num_labels_covered):  # Start from 1 to ignore the background
        # Get the coordinates of the connected component (covered cells)
        component_mask = (labels_covered == i).astype(np.uint8) * 255
        contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            # Draw a rectangle around the covered cell
            #cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green rectangle
            
            extract_bounding_box_image(image, x, y, w, h, "covered", i, templates)

    for i in range(1, num_labels_flagged):  # Start from 1 to ignore the background
        # Get the coordinates of the connected component (flagged cells)
        component_mask = (labels_flagged == i).astype(np.uint8) * 255
        contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            # Draw a rectangle around the flagged cell
            #cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red rectangle
            
            extract_bounding_box_image(image, x, y, w, h, "flagged", i, templates)

    # Show the image with bounding boxes around detected cells
    cv2.imshow("Detected Cells", image)

def main():
    templates = load_templates()
    while True:
        screen = capture_screen()  # Capture the screen
        detect_cells(screen, templates)  # Process the image and detect cells
        
        # Wait for a short period before capturing the next screen
        if cv2.waitKey(2) & 0xFF == ord('q'):  # Press 'q' to quit
            break

        # Remove the sleep for real-time detection
        time.sleep(5)  # Adjust this for faster or slower updates

    # Close the OpenCV window after exiting the loop
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

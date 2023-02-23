import cv2

from constants.paths import OUTPUT_DIR

WINDOW_NAME = "Image"

def show_image(image, window_name=WINDOW_NAME):
  cv2.imshow(window_name, image)
  # waits for user to press any key
  # (this is necessary to avoid Python kernel form crashing)
  cv2.waitKey(0)
    
  # closing all open windows
  cv2.destroyAllWindows()

def save_image(image, name = "output.png", path = OUTPUT_DIR):
  if not path.exists():
    path.mkdir(parents=True, exist_ok=False)
  
  path = str(path / name)
  cv2.imwrite(path, image)
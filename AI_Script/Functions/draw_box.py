import cv2
import numpy as np

# Load image
img = cv2.imread('/home/bht/CODE/ai-script/inputs/dataset/COCO_Val_2017_TEST/images/000000459757.jpg')

# Bounding box and class
boxes = np.array([[201, 191, 370, 309]])
cls = np.array([23])

# Draw rectangle
for box in boxes:
    x1, y1, x2, y2 = box
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img, f'Class {cls[0]}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# Save or display
cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
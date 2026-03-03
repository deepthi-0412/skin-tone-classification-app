import cv2
import numpy as np


def extract_skin(img):
    """
    Improved skin extraction:
    1. Detect face first
    2. Apply YCrCb skin segmentation
    3. Clean mask using morphology
    4. Keep largest skin region
    """

    if img is None:
        return img

    # Resize for consistency
    img = cv2.resize(img, (224, 224))

    # -----------------------------------
    # 1️⃣ FACE DETECTION FIRST
    # -----------------------------------
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    if len(faces) == 0:
        # No face detected → fallback to original
        return img

    # Take the largest detected face
    largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
    (x, y, w, h) = largest_face

    face = img[y:y + h, x:x + w]

    # -----------------------------------
    # 2️⃣ CONVERT TO YCrCb
    # -----------------------------------
    ycrcb = cv2.cvtColor(face, cv2.COLOR_BGR2YCrCb)

    # Skin color range
    lower = np.array([0, 133, 77], dtype=np.uint8)
    upper = np.array([255, 173, 127], dtype=np.uint8)

    mask = cv2.inRange(ycrcb, lower, upper)

    # -----------------------------------
    # 3️⃣ REMOVE NOISE
    # -----------------------------------
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    mask = cv2.erode(mask, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=2)
    mask = cv2.GaussianBlur(mask, (3, 3), 0)

    # -----------------------------------
    # 4️⃣ KEEP ONLY LARGEST SKIN REGION
    # -----------------------------------
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)
        clean_mask = np.zeros_like(mask)
        cv2.drawContours(clean_mask, [largest_contour], -1, 255, -1)
        mask = clean_mask

    skin = cv2.bitwise_and(face, face, mask=mask)

    # -----------------------------------
    # 5️⃣ SAFETY CHECK
    # -----------------------------------
    if np.count_nonzero(mask) < 500:
        # If segmentation failed → return face
        return face

    return skin
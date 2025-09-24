import cv2
import numpy as np


url = "http://10.166.228.60:8080/video"  
cap = cv2.VideoCapture(url)

# -------------------------
# Preprocessing
# -------------------------
def preprocess_image(frame):
    frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adaptive threshold
    thresh = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 6
    )

    # Morphological operations to remove noise
    kernel = np.ones((3, 3), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel, iterations=2)

    return morph

# -------------------------
# Particle detection
# -------------------------
def find_particles(thresh_img, min_area=20, max_area=5000):
    contours, _ = cv2.findContours(
        thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    particles = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area < area < max_area:
            perimeter = cv2.arcLength(cnt, True)
            circularity = 4 * np.pi * (area / (perimeter * perimeter + 1e-6))

            # Filter by shape (avoid random noise)
            if 0.3 < circularity < 1.2:
                x, y, w, h = cv2.boundingRect(cnt)
                particles.append({
                    'contour': cnt,
                    'area': area,
                    'bbox': (x, y, w, h)
                })
    return particles

# -------------------------
# Drawing results in um
# -------------------------
def draw_particles(frame, particles, mm_per_pixel=None):
    for p in particles:
        cv2.drawContours(frame, [p['contour']], -1, (0, 255, 0), 2)
        x, y, w, h = p['bbox']

        if mm_per_pixel:
            # Convert to micrometers
            um_per_pixel = mm_per_pixel * 1000
            width_um = w * um_per_pixel
            height_um = h * um_per_pixel
            area_um2 = p['area'] * (um_per_pixel ** 2)
            label = f"{width_um:.1f}x{height_um:.1f} um | {area_um2:.1f} um^2"
        else:
            label = f"Size(px): {int(p['area'])}"

        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    return frame

# -------------------------
# Average size calculation in um
# -------------------------
def calculate_average_size(particles, mm_per_pixel=None):
    if not particles:
        return None

    widths, heights, areas = [], [], []
    um_per_pixel = mm_per_pixel * 1000 if mm_per_pixel else None

    for p in particles:
        w, h = p['bbox'][2], p['bbox'][3]
        if um_per_pixel:
            widths.append(w * um_per_pixel)
            heights.append(h * um_per_pixel)
            areas.append(p['area'] * (um_per_pixel ** 2))
        else:
            widths.append(w)
            heights.append(h)
            areas.append(p['area'])

    avg_width = np.mean(widths)
    avg_height = np.mean(heights)
    avg_area = np.mean(areas)

    return avg_width, avg_height, avg_area

# -------------------------
# Calibration step
# -------------------------
# ⚠️ CHANGE THIS according to your reference object
KNOWN_WIDTH_MM = 20.0   # e.g., coin diameter in mm
REF_OBJECT_FOUND = False
mm_per_pixel = None

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Could not read from camera stream")
        break

    thresh = preprocess_image(frame)
    particles = find_particles(thresh)

    # Calibration: find the largest contour (assume it's reference object)
    if not REF_OBJECT_FOUND and particles:
        ref = max(particles, key=lambda p: p['area'])
        ref_w = ref['bbox'][2]  # bounding box width in pixels
        mm_per_pixel = KNOWN_WIDTH_MM / ref_w
        REF_OBJECT_FOUND = True
        print(f"Calibrated: {mm_per_pixel*1000:.2f} um/pixel")

    result_frame = draw_particles(frame.copy(), particles, mm_per_pixel)

    # -------------------------
    # Show average size
    # -------------------------
    avg_sizes = calculate_average_size(particles, mm_per_pixel)
    if avg_sizes:
        avg_width, avg_height, avg_area = avg_sizes
        if mm_per_pixel:
            avg_label = f"Avg: {avg_width:.1f}x{avg_height:.1f} um | {avg_area:.1f} um^2"
        else:
            avg_label = f"Avg Size(px): {avg_area:.1f}"

        cv2.putText(result_frame, avg_label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Show windows
    cv2.imshow('Microplastic Detection', result_frame)
    cv2.imshow('Preprocessed', thresh)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

import cv2
import numpy as np
import time
import zxingcpp
import csv
import os
from datetime import datetime

# --- Calibration Settings ---
# Known physical width of the QR code in millimeters (adjust this to your QR code's actual size)
KNOWN_WIDTH_MM = 50.0  

# Approximated Focal Length of camera (in pixels). 
FOCAL_LENGTH = 700.0  

def adjust_gamma(image, gamma=1.0):
    """
    Adjusts the brightness (gamma) of the image. 
    gamma < 1.0 darkens the image (helps with glare/overexposure).
    gamma > 1.0 brightens it.
    """
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def apply_clahe(image):
    """
    Applies Contrast Limited Adaptive Histogram Equalization.
    This enhances local contrast and helps resolve reflection/glare hotspots.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    equalized = clahe.apply(gray)
    return cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)

def main():
    global FOCAL_LENGTH
    
    # Open local webcam (usually 0 is built-in webcam)
    print("Opening webcam...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
        
    print("Webcam started. Press 'q' to exit. Press 'c' to calibrate focal length.")
    
    # CSV output configuration for Excel
    CSV_FILE = "scanned_products.csv"
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "QR_Data", "Distance_meters"])
            
    # Track the last successfully detected parameters
    detected_data = None
    pixel_width = 0.0
    pts = None
    last_detect_time = 0
    last_printed_data = None
    last_print_time = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame from webcam.")
            break
            
        current_time = time.time()
        
        # Decode using zxing-cpp
        barcodes = zxingcpp.read_barcodes(frame)
        
        # Glare/Brightness Pass 2: Apply CLAHE (Contrast Enhancement)
        if not barcodes:
            clahe_frame = apply_clahe(frame)
            barcodes = zxingcpp.read_barcodes(clahe_frame)
            
        # Glare/Brightness Pass 3: Darken image (useful if phone screen is too bright/glaring)
        if not barcodes:
            darkened_frame = adjust_gamma(frame, gamma=0.5)
            barcodes = zxingcpp.read_barcodes(darkened_frame)
        
        for barcode in barcodes:
            pos = barcode.position
            if pos:
                pts = np.array([
                    [pos.top_left.x, pos.top_left.y],
                    [pos.top_right.x, pos.top_right.y],
                    [pos.bottom_right.x, pos.bottom_right.y],
                    [pos.bottom_left.x, pos.bottom_left.y]
                ], dtype=np.int32)
                detected_data = barcode.text
                
                # Calculate pixel width of the QR code (average of top and bottom side lengths)
                top_width = np.linalg.norm(pts[0] - pts[1])
                bottom_width = np.linalg.norm(pts[2] - pts[3])
                pixel_width = (top_width + bottom_width) / 2.0
                last_detect_time = current_time
            
        # If detected recently, display details
        if pts is not None and (current_time - last_detect_time < 0.5):
            # Draw bounding box
            for i in range(4):
                cv2.line(frame, tuple(pts[i]), tuple(pts[(i+1)%4]), (0, 255, 0), 2)
                
            if pixel_width > 0:
                # Estimate distance (Depth) using pinhole camera model
                distance_mm = (KNOWN_WIDTH_MM * FOCAL_LENGTH) / pixel_width
                distance_meters = distance_mm / 1000.0
                
                display_text = f"Data: {detected_data[:30]}" if detected_data else "QR Detected"
                cv2.putText(frame, display_text, (pts[0][0], pts[0][1] - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                cv2.putText(frame, f"Size: {KNOWN_WIDTH_MM}mm | Pixels: {int(pixel_width)}px", 
                            (pts[0][0], pts[0][1] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                
                cv2.putText(frame, f"Depth: {distance_meters:.2f}m ({distance_mm:.1f}mm)", 
                            (pts[0][0], pts[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                # Print to terminal console
                if detected_data != last_printed_data or (current_time - last_print_time > 2.0):
                    print(f"\n[SCANNED DATA]: {detected_data}")
                    print(f"Distance: {distance_meters:.2f} meters ({distance_mm:.1f} mm)")
                    
                    # Append directly to CSV file
                    with open(CSV_FILE, mode='a', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), detected_data, f"{distance_meters:.2f}"])
                    print(f"--> Saved to {CSV_FILE}")
                    
                    last_printed_data = detected_data
                    last_print_time = current_time
        
        # Show image
        cv2.imshow("Webcam QR Scanner", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c') and pts is not None and (current_time - last_detect_time < 0.5):
            try:
                print("\n--- Calibration Mode ---")
                actual_dist_input = input("Enter current actual distance from webcam to QR code (in cm): ")
                actual_dist_mm = float(actual_dist_input) * 10.0
                
                # Calculate new focal length
                new_focal = (pixel_width * actual_dist_mm) / KNOWN_WIDTH_MM
                FOCAL_LENGTH = new_focal
                print(f"Calibrated successfully! New Focal Length set to: {FOCAL_LENGTH:.2f}\n")
            except Exception as e:
                print("Invalid input. Calibration skipped.", e)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

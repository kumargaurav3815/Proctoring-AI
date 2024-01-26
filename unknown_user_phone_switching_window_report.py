import face_recognition
import cv2
import tkinter as tk
import numpy as np
import pygetwindow as gw
import pyautogui
import time
import os


def take_screenshot(window_title):
    window = gw.getWindowsWithTitle(window_title)
    if window:
        window = window[0]
        screenshot = pyautogui.screenshot(region=(window.left, window.top, window.width, window.height))
        screenshot_path = "screenshot.png"
        screenshot.save(screenshot_path)
        return screenshot_path
    else:
        return None

def get_active_window_title():
    try:
        active_window = gw.getActiveWindow()
        return active_window.title if active_window else ""
    except Exception as e:
        print(f"Error getting active window title: {e}")
        return ""

known_image = face_recognition.load_image_file("./images/Photograph.jpeg")

known_face_encoding = face_recognition.face_encodings(known_image)[0]

known_face_encodings = [known_face_encoding]
known_face_labels = ["Known Person"]

net = cv2.dnn.readNet("./models/yolov3.cfg", './models/yolov3.weights')
classes = []
with open("./models/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getUnconnectedOutLayersNames()

video_capture = cv2.VideoCapture(0)

if not video_capture.isOpened():
    print("Error: Could not open webcam.")
    exit()

cv2.namedWindow("Video", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Video", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

root = tk.Tk()
root.withdraw()

user_detected = False
unknown_detected = False
last_active_window_title = get_active_window_title()
start_time = None

violations = {
    "Multiple or Unknown User": False,
    "Phone Detected": False,
    "Window Switched": False,
    "No User Detected": False
}

unknown_users = []  

while True:
    ret, frame = video_capture.read()

    if not ret:
        print("Error: Could not read frame.")
        break

    face_locations = face_recognition.face_locations(frame)

    if not face_locations:
        print("No face detected!")
        violations["No User Detected"] = True

    try:
        face_encodings = face_recognition.face_encodings(frame, face_locations)
    except Exception as e:
        print(f"Error computing face encodings: {e}")
        continue

    num_users = len(face_encodings)
    print(f"Users detected: {num_users}")

    if num_users > 1 or any(not any(face_recognition.compare_faces(known_face_encodings, face_encoding)) for face_encoding in face_encodings):
        violations["Multiple or Unknown User"] = True
        print("Unknown user detected!")

        for face_encoding in face_encodings:
            if not any(face_recognition.compare_faces(known_face_encodings, face_encoding)):
                unknown_users.append({
                    "encoding": face_encoding,
                    "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
                })

        for (top, right, bottom, left) in face_locations:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        cv2.destroyAllWindows()
        video_capture.release()

        disqualified_message = "Disqualified! Window switched"
        print(disqualified_message)

        report_content = f"Report generated at {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        for violation, detected in violations.items():
            if detected:
                report_content += f"\n- {violation}"

        if unknown_users:
            report_content += "\n\nUnknown Users:"
            for user in unknown_users:
                report_content += f"\n- Timestamp: {user['timestamp']}"

        report_filename = "report.txt"
        with open(report_filename, "w") as report_file:
            report_file.write(report_content)

        print(f"Automated report saved: {report_filename}")

        exit()

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        name = "Unknown"

        if any(matches):
            first_match_index = matches.index(True)
            name = known_face_labels[first_match_index]

        print("Recognized Name:", name)

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))

    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(layer_names)

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and classes[class_id] == "cell phone":
                print("Detected Phone:", classes[class_id], "Confidence:", confidence)
                violations["Phone Detected"] = True

                disqualified_message = "Disqualified! Phone detected"
                cv2.putText(frame, disqualified_message, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow("Video", frame)
                cv2.waitKey(5000) 
                screenshot_path = take_screenshot(get_active_window_title())
                if screenshot_path:
                    print(f"Screenshot saved: {screenshot_path}")

                report_content = f"Report generated at {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                for violation, detected in violations.items():
                    if detected:
                        report_content += f"\n- {violation}"

                if unknown_users:
                    report_content += "\n\nUnknown Users:"
                    for user in unknown_users:
                        report_content += f"\n- Timestamp: {user['timestamp']}"

                report_filename = "report.txt"
                with open(report_filename, "w") as report_file:
                    report_file.write(report_content)

                print(f"Automated report saved: {report_filename}")

                cv2.destroyAllWindows() 
                video_capture.release() 
                exit()  

    active_window_title = get_active_window_title()
    if active_window_title != last_active_window_title:
        start_time = time.time()
        last_active_window_title = active_window_title
        violations["Window Switched"] = True

        disqualified_message = "Disqualified! Window switched"
        print(disqualified_message)

        report_content = f"Report generated at {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        for violation, detected in violations.items():
            if detected:
                report_content += f"\n- {violation}"

        if unknown_users:
            report_content += "\n\nUnknown Users:"
            for user in unknown_users:
                report_content += f"\n- Timestamp: {user['timestamp']}"

        report_filename = "report.txt"
        with open(report_filename, "w") as report_file:
            report_file.write(report_content)

        print(f"Automated report saved: {report_filename}")

        cv2.destroyAllWindows()
        video_capture.release()
        exit() 

    cv2.imshow("Video", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()

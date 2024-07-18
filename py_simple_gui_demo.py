import cv2
import PySimpleGUI as sg

# Define the window layout
layout = [
    [sg.Image(filename='', key='image', size=(1920, 1080))],  # Adjust the size of the image element
    [sg.Button('Exit')]
]

# Create the window
window = sg.Window('OpenCV with PySimpleGUI', layout, location=(0, 0))

# Capture video from the webcam
cap = cv2.VideoCapture(0)

while True:
    event, values = window.read(timeout=20)
    if event == sg.WIN_CLOSED or event == 'Exit':
        break

    ret, frame = cap.read()
    if not ret:
        continue

    imgbytes = cv2.imencode('.png', frame)[1].tobytes()
    window['image'].update(data=imgbytes)

# Release the webcam and close the window
cap.release()
window.close()

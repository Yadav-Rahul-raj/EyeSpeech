#This project is created on Jetson-Nano kit(4 Gb)

#Write text with the help of eye detection and at last converting text in Audio form

#Disable to create __pycache__
import sys
sys.dont_write_bytecode = True

#importing library
import os
import cv2
import numpy as np
import dlib
from math import hypot
import pyglet
import time
from gtts import gTTS

# for CSI camera
def gstreamer_pipeline(
	capture_width=500,
	capture_height=500,
	display_width=500,
	display_height=500,
	framerate=30,
	flip_method=0,
):
	return (
		"nvarguscamerasrc ! "
		"video/x-raw(memory:NVMM), "
		"width=(int)%d, height=(int)%d, "
		"format=(string)NV12, framerate=(fraction)%d/1 ! "
		"nvvidconv flip-method=%d ! "
		"video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
		"videoconvert ! "
		"video/x-raw, format=(string)BGR ! appsink"
	      % (
			capture_width,
			capture_height,
			framerate,
			flip_method,
			display_width,
			display_height,
	  	)
  )


# Load sounds
sound = pyglet.media.load("sound.wav", streaming=False)
left_sound = pyglet.media.load("left.wav", streaming=False)
right_sound = pyglet.media.load("right.wav", streaming=False)


#cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER) #for CSI Camera
#cap = cv2.VideoCapture(1) #for external camera
cap = cv2.VideoCapture(0) #for internal camera
board = np.zeros((500, 600), np.uint8)
board[:] = 255

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Keyboard settings
keyboard = np.zeros((500, 500,3), np.uint8)
keys_set_1 = {0: "1", 1: "2", 2: "3", 3: "4", 4: "5",
              5: "Q", 6: "W", 7: "E", 8: "R", 9: "T",
              10: "A", 11: "S", 12: "D", 13: "F", 14: "G"
              ,15: "Z", 16: "X", 17: "C", 18: "V", 19: "-",
              20:"!",21:"@",22:"$",23:"EXT",24:"<"}

keys_set_2 = {0: "6", 1: "7", 2: "8", 3: "9", 4: "0",
              5: "Y", 6: "U", 7: "I", 8: "O", 9: "P",
              10: "H", 11: "J", 12: "K", 13: "L", 14: ","
              ,15: "B", 16: "N", 17: "M", 18: "'",19: ".",
              20:"_",21:"?",22:"(",23:")",24:"<"}

def draw_letters(letter_index, text, letter_light):
    # Keys
    if letter_index == 0:
        x = 0
        y = 0
    elif letter_index == 1:
        x = 100
        y = 0
    elif letter_index == 2:
        x = 200
        y = 0
    elif letter_index == 3:
        x = 300
        y = 0
    elif letter_index == 4:
        x = 400
        y = 0
    elif letter_index == 5:
        x = 0
        y = 100
    elif letter_index == 6:
        x = 100
        y = 100
    elif letter_index == 7:
        x = 200
        y = 100
    elif letter_index == 8:
        x = 300
        y = 100
    elif letter_index == 9:
        x = 400
        y = 100
    elif letter_index == 10:
        x = 0
        y = 200
    elif letter_index == 11:
        x = 100
        y = 200
    elif letter_index == 12:
        x = 200
        y = 200
    elif letter_index == 13:
        x = 300
        y = 200
    elif letter_index == 14:
        x = 400
        y = 200
    elif letter_index == 15:
        x = 0
        y = 300
    elif letter_index == 16:
        x = 100
        y = 300
    
    elif letter_index == 17:
        x = 200
        y = 300

    elif letter_index == 18:
        x = 300
        y = 300

    elif letter_index == 19:
        x = 400
        y = 300

    elif letter_index == 20:
        x = 0
        y = 400

    elif letter_index == 21:
        x = 100
        y = 400

    elif letter_index == 22:
        x = 200
        y = 400

    elif letter_index == 23:
        x = 300
        y = 400

    elif letter_index == 24:
        x = 400
        y = 400
        
    width = 100  #for key
    height = 100 #for key
    th = 3 # thickness

    # Text settings
    font_letter = cv2.FONT_HERSHEY_PLAIN
    font_scale = 2
    font_th = 4
    text_size = cv2.getTextSize(text, font_letter, font_scale, font_th)[0]
    width_text, height_text = text_size[0], text_size[1]
    text_x = int((width - width_text) / 2) + x   ## 2 is for the directoin of the text in the key
    text_y = int((height + height_text) / 2) + y

    if letter_light is True:
        cv2.rectangle(keyboard, (x + th, y + th), (x + width - th, y + height - th), (255, 0, 255), -1)
        cv2.putText(keyboard, text, (text_x, text_y), font_letter, font_scale, (51, 51, 51), font_th)
    else:
        cv2.rectangle(keyboard, (x + th, y + th), (x + width - th, y + height - th), (51, 51, 51), -1)
        cv2.putText(keyboard, text, (text_x, text_y), font_letter, font_scale, (255, 0, 255), font_th)

def draw_menu():
    rows, cols, _ = keyboard.shape    
    th_lines = 4 # thickness lines
    cv2.line(keyboard, (int(cols/2) - int(th_lines/2), 0),(int(cols/2) - int(th_lines/2), rows),
             (51, 51, 51), th_lines)
    cv2.putText(keyboard, "LEFT", (80, 250), font, 2, (255, 0, 255), 5)
    cv2.putText(keyboard, "RIGHT", (80 + int(cols/2), 250), font, 2, (255, 0, 255), 5)

def midpoint(p1 ,p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)

font = cv2.FONT_HERSHEY_PLAIN

def get_blinking_ratio(eye_points, facial_landmarks):
    left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
    right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
    center_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
    center_bottom = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))

    hor_line_lenght = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
    ver_line_lenght = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))

    ratio = hor_line_lenght / ver_line_lenght
    return ratio

def eyes_contour_points(facial_landmarks):
    left_eye = []
    right_eye = []
    for n in range(36, 42):
        x = facial_landmarks.part(n).x
        y = facial_landmarks.part(n).y
        left_eye.append([x, y])
    for n in range(42, 48):
        x = facial_landmarks.part(n).x
        y = facial_landmarks.part(n).y
        right_eye.append([x, y])
    left_eye = np.array(left_eye, np.int32)
    right_eye = np.array(right_eye, np.int32)
    return left_eye, right_eye

def get_gaze_ratio(eye_points, facial_landmarks):
    left_eye_region = np.array([(facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y),
                                (facial_landmarks.part(eye_points[1]).x, facial_landmarks.part(eye_points[1]).y),
                                (facial_landmarks.part(eye_points[2]).x, facial_landmarks.part(eye_points[2]).y),
                                (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y),
                                (facial_landmarks.part(eye_points[4]).x, facial_landmarks.part(eye_points[4]).y),
                                (facial_landmarks.part(eye_points[5]).x, facial_landmarks.part(eye_points[5]).y)], np.int32)

    height, width, _ = frame.shape
    mask = np.zeros((height, width), np.uint8)
    cv2.polylines(mask, [left_eye_region], True, 255, 2)
    cv2.fillPoly(mask, [left_eye_region], 255)
    eye = cv2.bitwise_and(gray, gray, mask=mask)

    min_x = np.min(left_eye_region[:, 0])
    max_x = np.max(left_eye_region[:, 0])
    min_y = np.min(left_eye_region[:, 1])
    max_y = np.max(left_eye_region[:, 1])

    gray_eye = eye[min_y: max_y, min_x: max_x]
    _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)
    height, width = threshold_eye.shape
    left_side_threshold = threshold_eye[0: height, 0: int(width / 2)]
    left_side_white = cv2.countNonZero(left_side_threshold)

    right_side_threshold = threshold_eye[0: height, int(width / 2): width]
    right_side_white = cv2.countNonZero(right_side_threshold)

    if left_side_white == 0:
        gaze_ratio = 1
    elif right_side_white == 0:
        gaze_ratio = 5
    else:
        gaze_ratio = left_side_white / right_side_white
    return gaze_ratio

# Counters (time )
frames = 0
letter_index = 0
blinking_frames = 0
frames_to_blink = 6
frames_active_letter = 5 

# Text and keyboard settings
text = ""
keyboard_selected = "left"
last_keyboard_selected = "left"
select_keyboard_menu = True
keyboard_selection_frames = 0

# converting the text into audio
def txtmp3(a):
    word = a
    fw = open("EyeSpeech.txt","w")
    fw.write(word)
    fw.flush()
    language = 'en'
    myobj = gTTS(text=word, lang=language, slow=False)
    myobj.save("EyeAudio.mp3")
    count =0
    while(count <3):
        wel = pyglet.media.load("EyeAudio.mp3", streaming=False)
        wel.play()
        time.sleep(2)
        count+=1


while True:
    _, frame = cap.read()
    #frame = cv2.resize(frame, None, fx=0.8, fy=0.8)
    rows, cols, _ = frame.shape
    keyboard[:] = (26, 26, 26)
    frames += 1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Draw a white space for loading bar
    frame[rows - 50: rows, 0: cols] = (255, 255, 255)

    if select_keyboard_menu is True:
        draw_menu()

    # Keyboard selected
    if keyboard_selected == "left":
        keys_set = keys_set_1
    else:
        keys_set = keys_set_2
    active_letter = keys_set[letter_index]

    # Face detection
    faces = detector(gray)
    for face in faces:
        landmarks = predictor(gray, face)

        left_eye, right_eye = eyes_contour_points(landmarks)

        # Detect blinking
        left_eye_ratio = get_blinking_ratio([36, 37, 38, 39, 40, 41], landmarks)
        right_eye_ratio = get_blinking_ratio([42, 43, 44, 45, 46, 47], landmarks)
        blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2

        # Eyes color
        cv2.polylines(frame, [left_eye], True, (0, 0, 255), 2)
        cv2.polylines(frame, [right_eye], True, (0, 0, 255), 2)


        if select_keyboard_menu is True:

            # Detecting gaze to select Left or Right keybaord
            gaze_ratio_left_eye = get_gaze_ratio([36, 37, 38, 39, 40, 41], landmarks)
            gaze_ratio_right_eye = get_gaze_ratio([42, 43, 44, 45, 46, 47], landmarks)
            gaze_ratio = (gaze_ratio_right_eye + gaze_ratio_left_eye) / 2

            if gaze_ratio <= 0.9:
                keyboard_selected = "right"
                keyboard_selection_frames += 1

                # If Kept gaze on one side more than 15 frames, move to keyboard
                if keyboard_selection_frames == 15:        
                    select_keyboard_menu = False
                    right_sound.play()

                    # Set frames count to 0 when keyboard selected
                    frames = 0
                    keyboard_selection_frames = 0
                if keyboard_selected != last_keyboard_selected:
                    last_keyboard_selected = keyboard_selected
                    keyboard_selection_frames = 0
            else:
                keyboard_selected = "left"
                keyboard_selection_frames += 1

                # If Kept gaze on one side more than 15 frames, move to keyboard
                if keyboard_selection_frames == 15:        
                    select_keyboard_menu = False
                    left_sound.play()

                    # Set frames count to 0 when keyboard selected
                    frames = 0
                if keyboard_selected != last_keyboard_selected:
                    last_keyboard_selected = keyboard_selected
                    keyboard_selection_frames = 0

        else:
            # Detect the blinking to select the key that is lighting up
            if blinking_ratio > 4:
                blinking_frames += 1
                frames -= 1

                # Show green eyes when closed
                cv2.polylines(frame, [left_eye], True, (0, 255, 0), 2)
                cv2.polylines(frame, [right_eye], True, (0, 255, 0), 2)

                # Typing letter
                if blinking_frames == frames_to_blink:
                    if active_letter != "<" and active_letter != "_":
                        text += active_letter
                    if active_letter == "_":
                        text += " "
                    if active_letter == "EXT":
                        txtmp3(a)
                        cap.release()
                        cv2.destroyAllWindows()
                    sound.play()
                    select_keyboard_menu = True
            else:
                blinking_frames = 0


    # Display letters on the keyboard
    if select_keyboard_menu is False:
        if frames == frames_active_letter:
            letter_index += 1
            frames = 0
        if letter_index == 25:
            letter_index = 0
        for i in range(25):
            if i == letter_index:
                light = True
            else:
                light = False
            draw_letters(i, keys_set[i], light)

    # Show the text we're writing on the board
    cv2.putText(board, text, (0, 30), font, 2, 0, 2)
    a = text

    # Blinking loading bar
    percentage_blinking = blinking_frames / frames_to_blink
    loading_x = int(cols * percentage_blinking)
    cv2.rectangle(frame, (0, rows - 50), (loading_x, rows), (51, 51, 51), -1)

    #pop up windows for camera,keyboard, board
    cv2.imshow("Camera", frame)
    cv2.imshow("Virtual keyboard", keyboard)
    cv2.imshow("Board", board)

    #If Enter key is pressed then the program will be terminated
    key = cv2.waitKey(1)
    if key == 13:
        break
    

#releasing camera and destroying all windows        
cap.release()
cv2.destroyAllWindows()

#function to convert text to mp3
txtmp3(a)   

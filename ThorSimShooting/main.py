from cvzone.HandTrackingModule import HandDetector
from cvzone.FaceDetectionModule import FaceDetector
import cv2
import math
import numpy as np
import random
import time

def main():
    # Set the dimensions of the window
    window_width = 800
    window_height = 600

    # Initialize the webcam to capture video
    # The '2' indicates the third camera connected to your computer; '0' would usually refer to the built-in camera
    cap = cv2.VideoCapture(0)
    count = 0

    lightning = cv2.VideoCapture("Arc.mov")
    arcShot = cv2.VideoCapture("ArcShot.mp4")

    # Check if the video file was opened successfully
    if not lightning.isOpened():
        print("Error: Could not open video file.")
        exit()

    # Get the codec information
    codec_info = lightning.get(cv2.CAP_PROP_FOURCC)

    # Convert the codec code to a human-readable format
    codec_fourcc = int(codec_info).to_bytes(4, 'little').decode('utf-8')

    print(f"Codec FourCC: {codec_fourcc}")

    # Release the video capture object
    lightning.release()

    # Initialize the HandDetector class with the given parameters
    detector_hand = HandDetector(staticMode=False,maxHands=2,modelComplexity=1,detectionCon=0.5,minTrackCon=0.5)
    detector_face = FaceDetector(minDetectionCon=0.5, modelSelection=0)
    # Continuously get frames from the webcam

    #Enemy กานต์

    # Load the enemy image
    enemy_image = cv2.imread('enemy.png', cv2.IMREAD_UNCHANGED)  # Replace 'path_to_your_image_file.png' with the actual file path
    enemy_image = cv2.resize(enemy_image, (100, 100))

    # Set initial positions (random x-coordinate, random y-coordinate not more than window_height // 2)
    x_position = 0
    # Initialize spawn_side variable
    spawn_side = random.choice(['left', 'right'])

    # Set initial movement direction based on spawn position
    x_speed = 5 if x_position < window_width // 2 else -5  # Positive if on left side, negative if on right side

    # Create a window
    # cv2.namedWindow('Thor simulator shooting', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('Thor simulator shooting', window_width, window_height)
    cv2.namedWindow('Thor simulator shooting', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Thor simulator shooting",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
    # ------------------------------------------------------------------------

    # Add a boolean variable to control the visibility of the enemy
    # enemy_visible = True
    # hide_enemy_time = 0
    show_enemy_duration = 30  # Set the duration in frames (e.g., 2 seconds at 30 frames per second)


    # Initialize a variable to keep track of the last time a hand was detected
    last_hand_detection_time = time.time()

    # Initialize a variable to store the countdown duration in seconds
    countdown_duration = 30

    # Initialize a variable to track if the countdown is active
    countdown_active = False

    score = 0
    time_remaining = 30

    textscore_time = True
    textresult = False
    result = 0

    while True:
        # Capture each frame from the webcam
        # 'success' will be True if the frame is successfully captured, 'img' will contain the frame
        success, img = cap.read()
        img = cv2.flip(img, 1)
        # Find hands in the current frame
        # The 'draw' parameter draws landmarks and hand outlines on the image if set to True
        # The 'flipType' parameter flips the image, making it easier for some detections
        hands, img = detector_hand.findHands(img, draw=False, flipType=True)
        # img, bboxs = detector_face.findFaces(img, draw=False)

        # if bboxs:
        #     # Loop through each bounding box
        #     for bbox in bboxs:
        #         # bbox contains 'id', 'bbox', 'score', 'center'

        #         # ---- Get Data  ---- #
        #         faceCenter = bbox["center"]
        #         x, y, w, h = bbox['bbox']
        #         score = int(bbox['score'][0] * 100)



        #         # ---- Draw Data  ---- #
        #         cv2.circle(img, faceCenter, 5, (255, 0, 255), cv2.FILLED)
        #         cvzone.putTextRect(img, f'{score}%', (x, y - 15),border=5)
        #         cvzone.cornerRect(img, (x, y, w, h))

        # enemy karn

    # Get the dimensions of the camera frame
        img_height, img_width, _ = img.shape

        # Calculate the width and height of the enemy image
        enemy_height, enemy_width, _ = enemy_image.shape

        if x_position + enemy_width >= img_width or x_position < 20:
            print("enemy spawn again")
            # Enemy image has hit the right side or left side, randomly choose a side to respawn
            spawn_side = random.choice(['left', 'right'])
            if spawn_side == 'left':
                # Respawn on the left side and move right
                print("Spawn on left, moving right")
                x_position = 20
            else:
                # Respawn on the right side and move left
                print("Spawn on right, moving left")
                x_position = img_width - enemy_width

            # Respawn with a random y-coordinate on the top side of the window
            y_position = random.randint(0, (img_height - enemy_height) // 2)

        # Move the enemy image left if it spawned on the right side
        # Move the enemy image right if it spawned on the left side
        if spawn_side == 'left':
            x_position = (x_position + 5) % (img_width + enemy_width)
        else:
            x_position = (x_position - 5) % (img_width + enemy_width)

        # Overlay the enemy image onto the frame at the calculated position
        for c in range(0, 3):
            img[y_position:y_position + enemy_height, x_position:x_position + enemy_width, c] = \
                img[y_position:y_position + enemy_height, x_position:x_position + enemy_width, c] * (1 - enemy_image[:, :, 3] / 255.0) + enemy_image[:, :, c] * (enemy_image[:, :, 3] / 255.0)



        # enemy karn

        # Check if any hands are detected
        if hands:
            # Information for the first hand detected
            hand1 = hands[0]  # Get the first hand detected
            lmList1 = hand1["lmList"]  # (x,y,z) List of 21 landmarks for the first hand
            bbox1 = hand1["bbox"]  # Bounding box around the first hand (x,y,w,h coordinates)
            center1 = hand1['center']  # Center coordinates of the first hand
            handType1 = hand1["type"]  # Type of the first hand ("Left" or "Right")


            # Count the number of fingers up for the first hand
            fingers1 = detector_hand.fingersUp(hand1)
            # print(f'H1 = {fingers1.count(1)}', end=" ")  # Print the count of fingers that are up
            if(fingers1.count(1) == 2) and textresult:
                textscore_time = True
                textresult = False
                score = 0  # Reset the score
                time_remaining = countdown_duration # Reset the time remaining
                
            tipOfIndexFinger = lmList1[8][0:2]
            tipOfMiddleFinger = lmList1[9][0:2]
            # # Calculate distance between specific landmarks on the first hand and draw it on the image
            # length, info, img = detector_hand.findDistance(tipOfIndexFinger,tipOfMiddleFinger , img, color=(255, 0, 255),scale=5)


            # Check if a second hand is detected
            if len(hands) == 2:
                # Information for the second hand
                hand2 = hands[1]
                lmList2 = hand2["lmList"]
                bbox2 = hand2["bbox"]
                center2 = hand2['center']
                handType2 = hand2["type"]

                # Count the number of fingers up for the second hand
                fingers2 = detector_hand.fingersUp(hand2)
                # print(f'H2 = {fingers2.count(1)}', end=" ")
                tipOfMiddleFinger2 = lmList2[9][0:2]
                # Calculate distance between the index fingers of both hands and draw it on the image

                # length, info, img = detector_hand.findDistance(tipOfMiddleFinger, tipOfMiddleFinger2, img, color=(255, 0, 0), scale=10)

                ret, frame_from_mp4 = lightning.read()
                ret2, arcShot_mp4 = arcShot.read()
                # Apply background subtraction to the frame
                if not ret:
                    # Reached the end of the video, so reopen the video file
                    lightning.release()  # Release the current video file
                    lightning = cv2.VideoCapture("Arc.mov")  # Reopen the video file
                    continue  # Continue to the next iteration of the loop

                if not ret2:
                    # Reached the end of the video, so reopen the video file
                    arcShot.release()  # Release the current video file
                    arcShot = cv2.VideoCapture("ArcShot.mp4")  # Reopen the video file
                    continue  # Continue to the next iteration of the loop

                # Calculate the center point of the line
                center_x = (tipOfMiddleFinger[0] + tipOfMiddleFinger2[0]) / 2
                center_y = (tipOfMiddleFinger[1] + tipOfMiddleFinger2[1]) / 2

                # Create a tuple representing the center point
                center = (int(center_x), int(center_y))

                # Calculate the slope of the line (check for division by zero)
                if tipOfMiddleFinger[0] - tipOfMiddleFinger2[0] != 0:
                    slope = (tipOfMiddleFinger[1] - tipOfMiddleFinger2[1]) / (tipOfMiddleFinger[0] - tipOfMiddleFinger2[0])
                else:
                    slope = float('inf')

                # Calculate the negative reciprocal of the slope to get the slope of the perpendicular line
                if slope != 0:
                    perpendicular_slope = -1 / slope
                else:
                    perpendicular_slope = float('inf')

                if (handType2 == "Left"):
                    if(tipOfMiddleFinger[1] <= tipOfMiddleFinger2[1]):
                        perpendicular_line_length = 1000
                    else:
                        perpendicular_line_length = -1000
                else:
                    if(tipOfMiddleFinger[1] <= tipOfMiddleFinger2[1]):
                        perpendicular_line_length = -1000
                    else:
                        perpendicular_line_length = 1000

                if perpendicular_slope != float('inf'):
                    perpendicular_endpoint_x = int(center_x + perpendicular_line_length * math.cos(math.atan(perpendicular_slope)))
                    perpendicular_endpoint_y = int(center_y + perpendicular_line_length * math.sin(math.atan(perpendicular_slope)))
                else:
                    perpendicular_endpoint_x = int(center_x)
                    perpendicular_endpoint_y = int(center_y + perpendicular_line_length)

                perpendicular_endpoint = ((perpendicular_endpoint_x), (perpendicular_endpoint_y))

                # Define the line color and thickness
                line_color = (0, 0, 255)  # Red color
                line_thickness = 1

                # Define the length of each dash and gap (in pixels)
                dash_length = 10  # You can adjust this value to change the dash length
                gap_length = 10  # You can adjust this value to change the gap length

                # Calculate the total length of the line
                line_length = int(cv2.norm(perpendicular_endpoint, center))

                # Initialize a variable to keep track of the current position
                current_pos = 0

                # Draw the dashed line
                while current_pos < line_length:
                    # Calculate the endpoints for the current dash segment
                    start_point = (int(center[0] + current_pos * (perpendicular_endpoint[0] - center[0]) / line_length),
                                int(center[1] + current_pos * (perpendicular_endpoint[1] - center[1]) / line_length))
                    current_pos += dash_length

                    # Ensure we don't exceed the line length
                    if current_pos > line_length:
                        current_pos = line_length

                    end_point = (int(center[0] + current_pos * (perpendicular_endpoint[0] - center[0]) / line_length),
                                int(center[1] + current_pos * (perpendicular_endpoint[1] - center[1]) / line_length))

                    # Draw the current dash segment
                    cv2.line(img, start_point, end_point, line_color, line_thickness)

                    # Move to the next position for the gap
                    current_pos += gap_length

                # Calculate the slope (m)
                if center[0] != perpendicular_endpoint[0]:
                    m = (center[1] - perpendicular_endpoint[1]) / (center[0] - perpendicular_endpoint[0])
                else:
                    m = float('inf')  # Handle the case of a vertical line

                if m != float('inf'):
                    # If the line is not vertical
                    b = center_y - m * center_x
                else:
                    b = float('inf')

                # Uncomment this code to check for overlap with the specified point
                # if (m != 0 and
                #     faceCenter[0] - 100 <= (faceCenter[1]-b)/m <= faceCenter[0] + 100 and
                #     faceCenter[1] - 100 <= m * faceCenter[0] + b <= faceCenter[1] + 100
                # and count%20 == 0):
                #     print(f"Overlapped with {faceCenter}")
                # elif(b==float('inf')):
                #     print("Error line is vertical")
                # else:
                #     if(count%20 == 0):
                #         print("<-------->")
                # Overlap object
                # get enemy data position
                if (m != 0 and
                    x_position - 200 <= (y_position-b)/m <= x_position + 200 and
                    y_position - 200 <= m * x_position + b <= y_position + 200


                and count%20 == 0):
                    print("enemy detect")
                    score += 1
                    print("enemy spawn again")
                    # Enemy image has hit the right side or left side, randomly choose a side to respawn
                    spawn_side = random.choice(['left', 'right'])
                    if spawn_side == 'left':
                        # Respawn on the left side and move right
                        print("Spawn on left, moving right")
                        x_position = 50
                    else:
                        # Respawn on the right side and move left
                        print("Spawn on right, moving left")
                        x_position = img_width - enemy_width

                    # Respawn with a random y-coordinate on the top side of the window
                    y_position = random.randint(0, (img_height - enemy_height) // 2)
                    

                elif(b==float('inf')):
                    print("Error line is vertical")
                else:
                    if(count%20 == 0):
                        print("<-------->")
                count += 1
                if not countdown_active and textscore_time:
                    last_hand_detection_time = time.time()  # Start the countdown
                    countdown_active = True
                
                
                

                if(handType2=="Right"):
                    # Resize the frame from the MP4 file to match the dimensions of the region where you want to overlay it
                    target_region = img[int(center_y)-int(math.sqrt(abs(tipOfMiddleFinger2[0]-tipOfMiddleFinger[0])**2+abs(tipOfMiddleFinger2[1]-tipOfMiddleFinger[1])**2)/2) > 0 and int(center_y)-int(math.sqrt(abs(tipOfMiddleFinger2[0]-tipOfMiddleFinger[0])**2+abs(tipOfMiddleFinger2[1]-tipOfMiddleFinger[1])**2)/2) or 1:int(center_y)+int(math.sqrt(abs(tipOfMiddleFinger2[0]-tipOfMiddleFinger[0])**2+abs(tipOfMiddleFinger2[1]-tipOfMiddleFinger[1])**2)/2), int(center_x)-int(math.sqrt(abs(tipOfMiddleFinger2[0]-tipOfMiddleFinger[0])**2+abs(tipOfMiddleFinger2[1]-tipOfMiddleFinger[1])**2)/2) > 0 and int(center_x)-int(math.sqrt(abs(tipOfMiddleFinger2[0]-tipOfMiddleFinger[0])**2+abs(tipOfMiddleFinger2[1]-tipOfMiddleFinger[1])**2)/2) or 1:int(center_x)+int(math.sqrt(abs(tipOfMiddleFinger2[0]-tipOfMiddleFinger[0])**2+abs(tipOfMiddleFinger2[1]-tipOfMiddleFinger[1])**2)/2)]



                    if(target_region.shape[1]>0 and target_region.shape[0] > 0):
                        frame_from_mp4 = cv2.resize(frame_from_mp4, (target_region.shape[1], target_region.shape[0]))
                        # Rotate the frame by a certain angle (e.g., 45 degrees)
                        rotation_angle = math.degrees(math.atan((tipOfMiddleFinger2[1]-tipOfMiddleFinger[1])/((tipOfMiddleFinger2[0]-tipOfMiddleFinger[0]) != 0 and (tipOfMiddleFinger2[0]-tipOfMiddleFinger[0]) or 0.0000001)))*-1  # Adjust the angle as needed
                        image_center = tuple(np.array(frame_from_mp4.shape[1::-1]) / 2)
                        rotation_matrix = cv2.getRotationMatrix2D(image_center, rotation_angle, 1)
                        frame_from_mp4 = cv2.warpAffine(frame_from_mp4, rotation_matrix, (frame_from_mp4.shape[1], frame_from_mp4.shape[0]), flags=cv2.INTER_LINEAR)

                        # Delete BG
                        lightning_gray = cv2.cvtColor(frame_from_mp4,cv2.COLOR_BGR2GRAY)
                        ret, mask = cv2.threshold(lightning_gray, 10, 255, cv2.THRESH_BINARY)
                        mask_inv = cv2.bitwise_not(mask)
                        lightning_fg = cv2.bitwise_and(frame_from_mp4,frame_from_mp4,mask = mask)
                        cap_bg = cv2.bitwise_and(target_region,target_region,mask = mask_inv)
                        dst = cv2.add(cap_bg,lightning_fg)

                        # Overlay the frame from the MP4 file on top of the current frame
                        img[int(center_y)-int(math.sqrt(abs(tipOfMiddleFinger2[0]-tipOfMiddleFinger[0])**2+abs(tipOfMiddleFinger2[1]-tipOfMiddleFinger[1])**2)/2) > 0 and int(center_y)-int(math.sqrt(abs(tipOfMiddleFinger2[0]-tipOfMiddleFinger[0])**2+abs(tipOfMiddleFinger2[1]-tipOfMiddleFinger[1])**2)/2) or 1:int(center_y)+int(math.sqrt(abs(tipOfMiddleFinger2[0]-tipOfMiddleFinger[0])**2+abs(tipOfMiddleFinger2[1]-tipOfMiddleFinger[1])**2)/2), int(center_x)-int(math.sqrt(abs(tipOfMiddleFinger2[0]-tipOfMiddleFinger[0])**2+abs(tipOfMiddleFinger2[1]-tipOfMiddleFinger[1])**2)/2) > 0 and int(center_x)-int(math.sqrt(abs(tipOfMiddleFinger2[0]-tipOfMiddleFinger[0])**2+abs(tipOfMiddleFinger2[1]-tipOfMiddleFinger[1])**2)/2) or 1:int(center_x)+int(math.sqrt(abs(tipOfMiddleFinger2[0]-tipOfMiddleFinger[0])**2+abs(tipOfMiddleFinger2[1]-tipOfMiddleFinger[1])**2)/2)] = dst

                    if(count%20>=0 and count%20<=5):
                        if(target_region.shape[1]>0 and target_region.shape[0]>0):
                            arcShot_mp4 = cv2.resize(arcShot_mp4, (target_region.shape[1], target_region.shape[0]))
                            arcShot_matrix = cv2.getRotationMatrix2D(image_center, rotation_angle+90, 1)
                            arcShot_mp4 = cv2.warpAffine(arcShot_mp4, arcShot_matrix, (arcShot_mp4.shape[1], arcShot_mp4.shape[0]), flags=cv2.INTER_LINEAR)

                            # Delete BG
                            arcShot_gray = cv2.cvtColor(arcShot_mp4,cv2.COLOR_BGR2GRAY)
                            arcret, arcmask = cv2.threshold(arcShot_gray, 10, 255, cv2.THRESH_BINARY)
                            arcmask_inv = cv2.bitwise_not(arcmask)
                            arc_fg = cv2.bitwise_and(arcShot_mp4,arcShot_mp4,mask = arcmask)
                            arccap_bg = cv2.bitwise_and(target_region,target_region,mask = arcmask_inv)
                            arcdst = cv2.add(arccap_bg,arc_fg)

                            img[int(center_y)-int(math.sqrt(abs(tipOfMiddleFinger2[0]-tipOfMiddleFinger[0])**2+abs(tipOfMiddleFinger2[1]-tipOfMiddleFinger[1])**2)/2) > 0 and int(center_y)-int(math.sqrt(abs(tipOfMiddleFinger2[0]-tipOfMiddleFinger[0])**2+abs(tipOfMiddleFinger2[1]-tipOfMiddleFinger[1])**2)/2) or 1:int(center_y)+int(math.sqrt(abs(tipOfMiddleFinger2[0]-tipOfMiddleFinger[0])**2+abs(tipOfMiddleFinger2[1]-tipOfMiddleFinger[1])**2)/2), int(center_x)-int(math.sqrt(abs(tipOfMiddleFinger2[0]-tipOfMiddleFinger[0])**2+abs(tipOfMiddleFinger2[1]-tipOfMiddleFinger[1])**2)/2) > 0 and int(center_x)-int(math.sqrt(abs(tipOfMiddleFinger2[0]-tipOfMiddleFinger[0])**2+abs(tipOfMiddleFinger2[1]-tipOfMiddleFinger[1])**2)/2) or 1:int(center_x)+int(math.sqrt(abs(tipOfMiddleFinger2[0]-tipOfMiddleFinger[0])**2+abs(tipOfMiddleFinger2[1]-tipOfMiddleFinger[1])**2)/2)] = arcdst
                else:
                    # Resize the frame from the MP4 file to match the dimensions of the region where you want to overlay it
                    target_region = img[int(center_y)-int(math.sqrt(abs(tipOfMiddleFinger[0]-tipOfMiddleFinger2[0])**2+abs(tipOfMiddleFinger[1]-tipOfMiddleFinger2[1])**2)/2) > 0 and int(center_y)-int(math.sqrt(abs(tipOfMiddleFinger[0]-tipOfMiddleFinger2[0])**2+abs(tipOfMiddleFinger[1]-tipOfMiddleFinger2[1])**2)/2) or 1:int(center_y)+int(math.sqrt(abs(tipOfMiddleFinger[0]-tipOfMiddleFinger2[0])**2+abs(tipOfMiddleFinger[1]-tipOfMiddleFinger2[1])**2)/2), int(center_x)-int(math.sqrt(abs(tipOfMiddleFinger[0]-tipOfMiddleFinger2[0])**2+abs(tipOfMiddleFinger[1]-tipOfMiddleFinger2[1])**2)/2) > 0 and int(center_x)-int(math.sqrt(abs(tipOfMiddleFinger[0]-tipOfMiddleFinger2[0])**2+abs(tipOfMiddleFinger[1]-tipOfMiddleFinger2[1])**2)/2) or 1:int(center_x)+int(math.sqrt(abs(tipOfMiddleFinger[0]-tipOfMiddleFinger2[0])**2+abs(tipOfMiddleFinger[1]-tipOfMiddleFinger2[1])**2)/2)]
                    if(target_region.shape[1]>0 and target_region.shape[0] > 0):
                        frame_from_mp4 = cv2.resize(frame_from_mp4, (target_region.shape[1], target_region.shape[0]))

                        # Rotate the frame by a certain angle (e.g., 45 degrees)
                        rotation_angle = math.degrees(math.atan((tipOfMiddleFinger[1]-tipOfMiddleFinger2[1])/((tipOfMiddleFinger2[0]-tipOfMiddleFinger[0]) != 0 and (tipOfMiddleFinger2[0]-tipOfMiddleFinger[0]) or 0.0000001)))  # Adjust the angle as needed
                        image_center = tuple(np.array(frame_from_mp4.shape[1::-1]) / 2)
                        rotation_matrix = cv2.getRotationMatrix2D(image_center, rotation_angle, 1)
                        frame_from_mp4 = cv2.warpAffine(frame_from_mp4, rotation_matrix, (frame_from_mp4.shape[1], frame_from_mp4.shape[0]), flags=cv2.INTER_LINEAR)

                        # Delete BG
                        lightning_gray = cv2.cvtColor(frame_from_mp4,cv2.COLOR_BGR2GRAY)
                        ret, mask = cv2.threshold(lightning_gray, 10, 255, cv2.THRESH_BINARY)
                        mask_inv = cv2.bitwise_not(mask)
                        lightning_fg = cv2.bitwise_and(frame_from_mp4,frame_from_mp4,mask = mask)
                        cap_bg = cv2.bitwise_and(target_region,target_region,mask = mask_inv)
                        dst = cv2.add(cap_bg,lightning_fg)

                        # Overlay the frame from the MP4 file on top of the current frame
                        img[int(center_y)-int(math.sqrt(abs(tipOfMiddleFinger[0]-tipOfMiddleFinger2[0])**2+abs(tipOfMiddleFinger[1]-tipOfMiddleFinger2[1])**2)/2) > 0 and int(center_y)-int(math.sqrt(abs(tipOfMiddleFinger[0]-tipOfMiddleFinger2[0])**2+abs(tipOfMiddleFinger[1]-tipOfMiddleFinger2[1])**2)/2) or 1:int(center_y)+int(math.sqrt(abs(tipOfMiddleFinger[0]-tipOfMiddleFinger2[0])**2+abs(tipOfMiddleFinger[1]-tipOfMiddleFinger2[1])**2)/2), int(center_x)-int(math.sqrt(abs(tipOfMiddleFinger[0]-tipOfMiddleFinger2[0])**2+abs(tipOfMiddleFinger[1]-tipOfMiddleFinger2[1])**2)/2) > 0 and int(center_x)-int(math.sqrt(abs(tipOfMiddleFinger[0]-tipOfMiddleFinger2[0])**2+abs(tipOfMiddleFinger[1]-tipOfMiddleFinger2[1])**2)/2) or 1:int(center_x)+int(math.sqrt(abs(tipOfMiddleFinger[0]-tipOfMiddleFinger2[0])**2+abs(tipOfMiddleFinger[1]-tipOfMiddleFinger2[1])**2)/2)] = dst

                    if(count%20>=0 and count%20<=5):
                        if(target_region.shape[1]>0 and target_region.shape[0]>0):
                            arcShot_mp4 = cv2.resize(arcShot_mp4, (target_region.shape[1], target_region.shape[0]))
                            arcShot_matrix = cv2.getRotationMatrix2D(image_center, rotation_angle+90, 1)
                            arcShot_mp4 = cv2.warpAffine(arcShot_mp4, arcShot_matrix, (arcShot_mp4.shape[1], arcShot_mp4.shape[0]), flags=cv2.INTER_LINEAR)

                            # Delete BG
                            arcShot_gray = cv2.cvtColor(arcShot_mp4,cv2.COLOR_BGR2GRAY)
                            arcret, arcmask = cv2.threshold(arcShot_gray, 10, 255, cv2.THRESH_BINARY)
                            arcmask_inv = cv2.bitwise_not(arcmask)
                            arc_fg = cv2.bitwise_and(arcShot_mp4,arcShot_mp4,mask = arcmask)
                            arccap_bg = cv2.bitwise_and(target_region,target_region,mask = arcmask_inv)
                            arcdst = cv2.add(arccap_bg,arc_fg)

                            img[int(center_y)-int(math.sqrt(abs(tipOfMiddleFinger[0]-tipOfMiddleFinger2[0])**2+abs(tipOfMiddleFinger[1]-tipOfMiddleFinger2[1])**2)/2) > 0 and int(center_y)-int(math.sqrt(abs(tipOfMiddleFinger[0]-tipOfMiddleFinger2[0])**2+abs(tipOfMiddleFinger[1]-tipOfMiddleFinger2[1])**2)/2) or 1:int(center_y)+int(math.sqrt(abs(tipOfMiddleFinger[0]-tipOfMiddleFinger2[0])**2+abs(tipOfMiddleFinger[1]-tipOfMiddleFinger2[1])**2)/2), int(center_x)-int(math.sqrt(abs(tipOfMiddleFinger[0]-tipOfMiddleFinger2[0])**2+abs(tipOfMiddleFinger[1]-tipOfMiddleFinger2[1])**2)/2) > 0 and int(center_x)-int(math.sqrt(abs(tipOfMiddleFinger[0]-tipOfMiddleFinger2[0])**2+abs(tipOfMiddleFinger[1]-tipOfMiddleFinger2[1])**2)/2) or 1:int(center_x)+int(math.sqrt(abs(tipOfMiddleFinger[0]-tipOfMiddleFinger2[0])**2+abs(tipOfMiddleFinger[1]-tipOfMiddleFinger2[1])**2)/2)] = arcdst
        
        
        # if enemy_visible:
            # Set the visibility flag to False
            # enemy_visible = False

            # Set the timer to show the enemy after a certain duration
            # hide_enemy_time = show_enemy_duration
        
        

        # Calculate the time remaining in the countdown
        time_elapsed = time.time() - last_hand_detection_time
        
        # If the enemy is not visible and the timer reaches zero, show the enemy
        # if not enemy_visible:
        #     enemy_visible = True
        
        if countdown_active:
            time_remaining = max(countdown_duration - time_elapsed, 0)
        
            # Check if the countdown timer has reached zero
            if time_remaining == 0:
                countdown_active = False
                result = score
                textscore_time = False
                textresult = True

        
        if textscore_time:
            # Display the countdown timer on the screen
            text_position = (330, window_height - 150)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1  # Font size
            font_color = (255, 255, 255)  # White font color
            font_thickness = 2
            countdown_text = f"Time remaining: {int(time_remaining)}"
            cv2.putText(img, countdown_text, text_position, font, font_scale, font_color, font_thickness, cv2.LINE_AA)

            # Decrement the hide_enemy_time if it's active
            # if not enemy_visible and hide_enemy_time > 0:
            #     hide_enemy_time -= 1
            text_position = (10, window_height - 150)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1  # Increased font size
            font_color = (255, 255, 255)  # White font color
            font_thickness = 2
            score_text = f"Score: {int(score)}"
            cv2.putText(img, score_text, text_position, font, font_scale, font_color, font_thickness, cv2.LINE_AA)
        
        if textresult:
            # Decrement the hide_enemy_time if it's active
            text_position = (100, window_height-350)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 2  # Increased font size
            font_color = (255, 255, 255)  # White font color
            font_thickness = 4
            score_text = f"Final Score: {int(result)}"
            cv2.putText(img, score_text, text_position, font, font_scale, font_color, font_thickness, cv2.LINE_AA)
            restart_text = f"Two-Finger up to restart"
            text_position = (120, window_height-450)
            cv2.putText(img, restart_text, text_position, font, 1, font_color, font_thickness, cv2.LINE_AA)

        # Show the img
        cv2.imshow('Thor simulator shooting', img)

        # Display the image in a window
        # cv2.imshow("Image", img)

        # Keep the window open and update it for each frame; wait for 1 millisecond between frames
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

    cv2.destroyAllWindows()
main()
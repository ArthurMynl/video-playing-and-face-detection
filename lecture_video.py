import cv2
import face_recognition 

# preprocessing
video_lecture = cv2.VideoCapture('video.mp4')
check , vid = video_lecture.read()

counter = 0
check = True
frame_list = []

while(check == True):
    cv2.imwrite("frame%d.jpg" %counter , vid)
    check , vid = video_lecture.read()
    frame_list.append(vid)
    counter += 1
    print(counter)
    
frame_list.pop()


face_locations = []
process_this_frame = True
cpt = 1
reverse, speed = False, False

video_capture = cv2.VideoCapture(0)

while(True):
    # calcul du nombre de personnes
    ret, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    rgb_small_frame = small_frame[:, :, ::-1]
    if process_this_frame:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        nb_faces = len(face_locations)
    process_this_frame = not process_this_frame
    
    # debug
    print(nb_faces)

    if(nb_faces > 5):
        reverse = True
        speed = False
    elif(nb_faces < 5):
        reverse = False
        speed = True
    else:
        reverse = False
        speed = False

    # gestion du défilement des images
    if(speed):
        cpt += 2
    elif(reverse):
        cpt -= 1
    else:
        cpt += 1
    
    if(cpt <= 0):
        cpt = len(frame_list) - 1
    elif(cpt >= len(frame_list)):
        cpt = 1
    
    cv2.imshow('video', frame_list[cpt])

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()

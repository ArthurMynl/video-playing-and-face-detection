import cv2
import face_recognition 
from tqdm import tqdm 

# getting the path of the video to display
with open('config.txt', 'r') as f:
    path = f.readline().split(" ")[-1]


# initialization of few variables
counter = 0
check = True
frame_list = []
face_locations = []
process_this_frame = True
cpt = 1
reverse, speed = False, False

# preprocessing of the video
video_lecture = cv2.VideoCapture(path)
check , vid = video_lecture.read()
frame_count = int(video_lecture.get(cv2.CAP_PROP_FRAME_COUNT))

for i in tqdm(range(frame_count), desc="Preprocessing"):
    if check == True:
        cv2.imwrite("frame%d.jpg" %counter , vid)
        check , vid = video_lecture.read()
        frame_list.append(vid)
        counter += 1
    
frame_list.pop()




video_capture = cv2.VideoCapture(0)

while(True):
    # calculation of the number of persons before the screen
    ret, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    rgb_small_frame = small_frame[:, :, ::-1]
    if process_this_frame:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        nb_faces = len(face_locations)
    process_this_frame = not process_this_frame
    
    # configuration of the variables to play the video
    if(nb_faces > 5):
        reverse = True
        speed = False
    elif(nb_faces < 5):
        reverse = False
        speed = True
    else:
        reverse = False
        speed = False

    # incrementation or decrementation of the counter to display the video
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

    # q or ctrl + c to stop the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()

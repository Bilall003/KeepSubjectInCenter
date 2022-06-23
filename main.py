import os
from tkinter.filedialog import askopenfilename
from tkinter import Tk
import shutil
import mediapipe as mp
import cv2
import PIL.Image as Image
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates
import numpy
import moviepy.editor as mvp

class centerObject():

    def __init__(self,path):
        print('Starting the app.....')
        self.frame1 = ''
        self.frame2 = ''
        self.start_x = 0
        self.end_x = 0
        self.position = ''

        self.old_fps = self.old_vide_frames(path)

        print('FPS of actual video '+str(self.old_fps))


        self.vid = cv2.VideoCapture(path)
        self.width  = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.mid_pt = self.width/2

        self.background_image = Image.new("RGB", (int(self.width), int(self.height)), "rgb(0, 0, 0)")
        self.background_image.save('background_image.png')
        self.background_image = Image.open('background_image.png')


        self.motion_frame = ''

        self.faceDetector = mp.solutions.face_detection
        self.drawing = mp.solutions.drawing_utils
        
        self.run(path)

    def motion_detection(self,f1,f2,draw=False):
        diff = cv2.absdiff(f1, f2)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(thresh, None, iterations=3)
        contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            
            if cv2.contourArea(contour) > 16000:

                (x,y,w,h)=cv2.boundingRect(contour)

                if (x+w)-(y+h) > 400:
                    self.motion_frame = self.frame1[y:y+h+100,x:x+w+200]

                    if draw:
                        cv2.rectangle(self.frame1,(x,y),(x+w,y+h),(0,255,0),2)

                    return True
        
        
        return False

    def crop_stage_area(self,f):
        f = f[80:600,100:1350]
        return f

    def detect_human(self,frame,draw=False):
        with self.faceDetector.FaceDetection(min_detection_confidence=0.5) as face_detection:
            
            # Convert the BGR image to RGB.
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            frame.flags.writeable = False
            results = face_detection.process(frame)

            # Draw the face detection annotations on the image.
            frame.flags.writeable = True
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


            image_rows, image_cols, _ = self.frame1.shape

            detected = results.detections

            if detected:
                try:
                    for d in detected:
                        location = d.location_data

                        relative_bounding_box = location.relative_bounding_box
                        rect_start_point = _normalized_to_pixel_coordinates(
                            relative_bounding_box.xmin, relative_bounding_box.ymin, image_cols,
                            image_rows)
                        rect_end_point = _normalized_to_pixel_coordinates(
                            relative_bounding_box.xmin + relative_bounding_box.width,
                            relative_bounding_box.ymin + relative_bounding_box.height, image_cols,
                            image_rows)

                        (x,_) = rect_start_point
                        (end_x,end_y) = rect_end_point

                        if end_x-end_y > 400:
                            self.start_x = x
                            self.end_x = end_x

                            for _, detection in enumerate(results.detections):
                    
                                if draw:
                                    # cv2.rectangle(self.frame1,rect_start_point,rect_end_point,(0,255,0),2)
                                    self.drawing.draw_detection(self.frame1, detection)


                            return True

                except:
                    pass
            
            return False

    def put_frame_on_black(self,frame):
        _, width, _ = frame.shape

        frame = Image.fromarray(frame)

        if self.position == 'RIGHT':
            self.background_image = Image.new("RGB", (int(self.width), int(self.height)), "rgb(0, 0, 0)")
            self.background_image.paste(frame, (0,0))
        
        else:
            self.background_image = Image.new("RGB", (int(self.width), int(self.height)), "rgb(0, 0, 0)")
            start_pt = self.width - width
            start_pt = int(start_pt)
            start_pt = abs(start_pt)

            self.background_image.paste(frame,(start_pt,0))

        frame = numpy.array(self.background_image)
        return frame

    def old_vide_frames(self,path):

        # Start default camera
        video = cv2.VideoCapture(path)

        
        # Find OpenCV version
        (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

        if int(major_ver)  < 3 :
            fps = video.get(cv2.cv.CV_CAP_PROP_FPS)

        else :
            fps = video.get(cv2.CAP_PROP_FPS)

        # Release video
        video.release()

        return fps

    def process_video(self):
        
        crop_x1 = 0
        crop_y1 = int(self.height)
        crop_x2 = 0
        crop_y2 = int(self.width)

        fps = self.old_fps/2

        video = cv2.VideoWriter('testvideo.avi',cv2.VideoWriter_fourcc('X','V','I','D'),fps, (int(self.width), int(self.height)))
        
        while self.vid.isOpened():
            _, self.frame1 = self.vid.read()
            _, self.frame2 = self.vid.read()
            

            f1 = self.crop_stage_area(self.frame1)
            f2 = self.crop_stage_area(self.frame2)

            #Check if there is any motion
            detect_motion = self.motion_detection(f1,f2)

            if detect_motion:
                if self.detect_human(self.motion_frame) and self.end_x != 0 and self.start_x != 0:

                    mid = (self.start_x+self.end_x)/2
                    
                    final_pt = self.mid_pt-mid

                    if final_pt<0:
                        final_pt = -1*final_pt
                        
                        crop_x2 = int(final_pt)
                        crop_y2 = int(self.width)
                        self.position = 'RIGHT'
                    else:
                        self.position = 'LEFT'
                        crop_x2 = 0
                        crop_y2 = int(self.width-final_pt)

                    print(self.position)

                    
            self.frame1 = self.frame1[crop_x1:crop_y1,crop_x2:crop_y2]
            self.frame1 = self.put_frame_on_black(self.frame1)

            video.write(self.frame1)
            
            cv2.imshow('Test',self.frame1)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.vid.release()
        video.release()
        cv2.destroyAllWindows()

    def extract_audio(self,path):
        my_clip = mvp.VideoFileClip(path)
        
        if my_clip.audio == None:
            return False

        my_clip.audio.write_audiofile("temp_audio.mp3")
        return True

    def combine_audio(self,vidname):
        
        fps = self.old_fps/2
        audname = 'temp_audio.mp3'
        my_clip = mvp.VideoFileClip(vidname)
        audio_background = mvp.AudioFileClip(audname)
        final_clip = my_clip.set_audio(audio_background)
        final_clip.write_videofile(vidname,fps=fps)

    def run(self,path):
        try:
            self.process_video()
        except:
            pass
        
        if self.extract_audio(path):
            self.combine_audio('testvideo.avi')
        else:
            shutil.copy('testvideo.avi',path)

        try:
            os.remove('testvideo.avi')
            os.remove('temp_audio.mp3')
        except:
            pass

        print('----------------Final video completed------------')

if __name__ == '__main__':
    root = Tk()
    pathh = askopenfilename()
    root.destroy()

    obj =  centerObject(pathh)
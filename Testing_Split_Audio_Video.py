from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.editor import VideoFileClip
import os
import mediapipe as mp
import numpy as np 
import cv2

def get_facial_landmarks_only(*args):
    frame=args[0]
    flag=args[1]
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh()
    black_img = np.zeros([frame.shape[0],frame.shape[1],3], dtype=np.uint8)
    # Show the frame (optional)
    results = face_mesh.process(frame)
    # Draw the facial landmarks on the image
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(image=black_img, landmark_list=face_landmarks, 
                                    landmark_drawing_spec=drawing_spec,
                                    connection_drawing_spec=drawing_spec)
    if flag==1:
        cv2.imwrite(args[2],black_img)
     
    
    return black_img
    

def save_frame_wise_embeddings(clip,name):

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = clip.fps
    width, height = clip.size

    iter=0
    for frame in clip.iter_frames():
        frame=get_facial_landmarks_only(frame,1,name+'_'+str(iter)+'.jpg')
        cv2.imshow('Whatup',frame)
        iter+=1
        if cv2.waitKey(1) == ord('q'):
            break





# Set up the drawing tools
mp_drawing = mp.solutions.drawing_utils 
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

def make_folder_dataset(name_of_vid):
    try:
        os.mkdir('Dataset')
    except:
        pass

    try:
        os.mkdir(f'Dataset/{name_of_vid}')
    except:
        pass

def get_time_stamps(name):
    name=name.split('.')[0]+'.txt'
    print(name)
    with open(name,'r') as f:
        texts=f.read()
    texts=texts.split('\n')
    return texts

def cut_video_frames(name_of_video):
    # Load the video clip
    video_clip = VideoFileClip(name_of_video)
    time_frame=get_time_stamps(name_of_video)
    name_of_video=(name_of_video.split('\\')[-1]).split('.')[0]
    make_folder_dataset(name_of_video)
    for indx,chunk in enumerate(time_frame):
        start_time,end_time=chunk.split('-')        
        video_clip_cut = video_clip.subclip(start_time, end_time).without_audio()
        audio_clip = video_clip.subclip(start_time, end_time).audio
        save_frame_wise_embeddings(video_clip_cut,f'Dataset/{name_of_video}/video_{indx}')
        video_clip_cut.write_videofile(f'Dataset/{name_of_video}/video_{indx}.mp4')
        audio_clip.write_audiofile(f'Dataset/{name_of_video}/dialogue_{indx}.wav')
    # close the video reader
    video_clip.reader.close()
    video_clip.audio.reader.close_proc()

#cut_video_frames('test_alpha.mp4')

if __name__=='__main__':
    path_of_input=input('Enter path of the file: ')
    for file in os.listdir(path_of_input):
        print(file)
        if file.endswith('.mp4'):
            cut_video_frames(path_of_input+'\\'+file)
    #print("Hello World")
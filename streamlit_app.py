import streamlit as st
import cv2
import torch
import tempfile
import os
import pandas as pd
import mediapipe as mp
import numpy as np
import mediapipe as mp
import sys
from streamlit_player import st_player
sys.path.append(
    r"C:\Users\joann\OneDrive - Robert Gordon University\Project\TD-GCN-Gesture")
sys.path.append(r"C:\Users\joann\OneDrive - Robert Gordon University\Project\sign-language-recognition\03_model_training")
from models.mobilenet_v4_hybrid_medium_model import MobilenetV5HybridMediumModel
from PIL import Image
from torchvision import transforms
from model.tdgcn import Model
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import shutil

labels = [
    "Book",
    "Car",
    "Eat",
    "Food",
    "Home",
    "House",
    "No",
    "Sit",
    "Toilet",
    "What",
    "Where",
    "Yes"
]
label1 = [
    "Book",
    "Car",
    "Eat",
    "Food",
    "Home",
    "House"]
label2=[    
    "No",
    "Sit",
    "Toilet",
    "What",
    "Where",
    "Yes"
]
example_path=(r"C:\Users\joann\OneDrive - Robert Gordon University\Project\Example Signs")
st.title("Makaton Recognition")
st.write("This model recognises Makaton Signs. With an emphasis on recognising signs from people with poor dexterity.")
st.subheader("It can recognise 12 signs: ")
col1, col2=st.columns(2)
with col1:
    for sign in label1:
            with st.expander(sign):
                eg_path=os.path.join(example_path, f"{sign}.gif")
                if os.path.exists(eg_path):
                     st.image(eg_path)
                else:
                    st.error(f"Missing video: {sign}")
with col2:
    for sign in label2:
        with st.expander(sign):
            eg_path=os.path.join(example_path, f"{sign}.gif")
            if os.path.exists(eg_path):
                 st.image(eg_path)
            else:
                st.error(f"Missing video: {sign}")

# "It uses MediaPipe to extract skeletal landmarks, Skeleton-DML as the image recognition algorithm, and MobileNetV4 Hybrid Medium as the  Upload a video of someone signing one of these signs and try it out for yourself!"
st.write("This demo has 2 different Models, MobileNetV4 based on CNN architecture and TD-GCN based on GCN architecture, both of which received the same accuracy. They both use MediaPipe to extract skeletal landmarks as shown below.")
with st.expander("How MediaPipe Extracts Landmarks"):
    st.write("""
    The left video shows the original video of a person signing Eat.
    The right video shows how MediaPipe extracts key points from the skeleton to track movement.
    """)
    col1, col2=st.columns(2)
    with col1:
        st.subheader("Original Video")
        st.image(r"C:\Users\joann\OneDrive - Robert Gordon University\Project\Original Video.gif")
    with col2:
        st.subheader("Processed Video")

        st.image(r"C:\Users\joann\OneDrive - Robert Gordon University\Project\skeleton.gif")
col1, col2=st.columns(2)
with col1:
    st.subheader("MobileNetV4 Metrics")
    st.metric( "MobileNetV4 Accuracy", "57.32%")
    st.metric( "MobileNetV4 F1 Score", "55.04%")
    st.image(r"C:\Users\joann\OneDrive - Robert Gordon University\Project\sign-language-recognition\99_model_output\results\8\mediapipe\2026-08-21_20-26-26_confusion_matrix.png", use_container_width=True)

with col2:
    st.subheader("TD-GCN Metrics")
    st.metric( "TD-GCN Accuracy", "57.32%")
    st.metric("TD-GCN F1 Score", "57.92%")
    st.image(r"C:\Users\joann\OneDrive - Robert Gordon University\Project\tdgcn_confusion_matrix.png")
st.write("""These models achieve the same accuracy but give different results and F1-scores. 
The 'best' recognised signs are based on precision not recall, which means how good the model is at correctly predicting True Positives, with no regard to incorrect predictions - False Positives. 
This can be seen in the MobileNetV4 model where one of the 'best' signs is Home as all Home signs were predicted as Home, but the model over predicts Home, classifying more Yes signs as Home than Yes.
While the TD-GCN model does not over predict any one sign, its errors are more random. 
""")
col1, col2=st.columns(2)
with col1:
    st.info(
    """
    Best recognised signs:
        
    • Home (100%)
        
    • Where (100%)
        
    • What (75%)

    Most challenging signs:
       
    • Yes (10%)
       
    • Book (35%)
        
    • Car (45%)
        
    """
    )
with col2:
    st.info(
    """
    Best recognised signs:

    • Where (82%)

    • Book (75%)

    • House (73%)

    Most challenging signs:
        
    • Sit (41%)
        
    • Home (43%)
        
    • Eat (47%)
    """
    )
input_method=st.radio("Choose Input Method", [
    "Upload Video",
    "Record Using Webcam"
    ])
#load model
WEIGHTS_PATH = r"C:\Users\joann\OneDrive - Robert Gordon University\Project\sign-language-recognition\99_model_output\results\8\mediapipe\models\2026-08-21_20-26-26.pth"
GCN_WEIGHTS = r"C:\Users\joann\OneDrive - Robert Gordon University\Project\TD-GCN-Gesture\work_dir\mediapipe_video_split\runs-25-1700.pt"

if "recording" not in st.session_state:
    st.session_state.recording=False
if "classified" not in st.session_state:
    st.session_state.classified=False 
video_path=None
if "video_path" not in st.session_state:
    st.session_state.video_path=None
if "feedback_submitted" not in st.session_state:
    st.session_state.feedback_submitted=False
@st.cache_resource 
def load_SLRmodel():
    base_SLRmodel=MobilenetV5HybridMediumModel(12)
    SLRmodel=base_SLRmodel.get_model()
    SLRmodel.load_state_dict(
        torch.load(WEIGHTS_PATH, map_location="cpu"))
    SLRmodel.eval()
    return SLRmodel, base_SLRmodel

SLRmodel, base_SLRmodel=load_SLRmodel()

@st.cache_resource
def load_GCNmodel():
    GCNmodel=Model(num_class=12,
                   num_point=67,
                   num_person=1,
                   graph="graph.mediapipe_sign.Graph",
                   graph_args={"labeling_mode":"spatial"})
    weights=torch.load(GCN_WEIGHTS, map_location="cpu")
    if "model_state_dict" in weights:
        weights=weights["model_state_dict"]
    GCNmodel.load_state_dict(weights, strict=False)
    GCNmodel.eval()
    return GCNmodel
GCNmodel=load_GCNmodel()
#extract MediaPipe Landmarks

# recorded_frames=[]
class VideoRecorder(VideoProcessorBase):
    def __init__(self):
        self.frames=[]
    def recv(self, frame):
        img=frame.to_ndarray(format="bgr24")
        self.frames.append(img)
        
        if len(self.frames) > 100:
            self.frames.pop(0)
        print(f"Frames stored: {len(self.frames)}")
        return av.VideoFrame.from_ndarray(img, format="bgr24")

def extract_video_data_from_path(video_path):
    df, original_gif_path,gif_path=process_video(video_path)
    return df, original_gif_path, gif_path

def extract_video_data(uploaded_file):
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".mp4") as tmp:
            file_bytes = uploaded_file.getvalue()
            tmp.write(file_bytes)
            video_path = tmp.name
    try:
        df, original_gif_path,gif_path =process_video(video_path)
    finally:
        try:
            os.remove(video_path)
        except:
            pass
    return df, original_gif_path, gif_path

def process_video(video_path):
    cap=cv2.VideoCapture(video_path)
    frame_number=0
    rows = []
    gif_frames=[]
    original_frames=[]
    gif_path=os.path.join(tempfile.gettempdir(), "skeleton_output.gif")
    mp_pose = mp.solutions.pose
    mp_hands = mp.solutions.hands
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing=mp.solutions.drawing_utils
    FACE_INDICES =[10,54,58,67,93,127,150,152,
    172,176,284,297,288,323,356,378,
    379,397,400,454,46,52,53,55,
    65,276,282,283,285,295,33,133,
    144,153,157,158,159,160,161,163,
    263,362,373,380,384,385,386,387,
    388,390,1,168,
    197,0,13,14,17,37,39,61,81,84,88,178,267,269,291,311,314, 324]

    with mp_pose.Pose() as pose, \
        mp_hands.Hands(max_num_hands=2) as hands,\
        mp_face_mesh.FaceMesh(static_image_mode=False,
                              max_num_faces=1,
                              refine_landmarks=True,
                              min_detection_confidence=0.5,
                              min_tracking_confidence=0.5) as face_mesh:
        while cap.isOpened():
            success, frame=cap.read()
            if not success:
                break
            # annotated_frame=frame.copy() #to get it on video
            annotated_frame=np.zeros_like(frame)
            row={"frame": frame_number}
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            original = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            original=cv2.resize(original, (640,360))
            original_frames.append(Image.fromarray(original))
            pose_result = pose.process(rgb)
            hand_result = hands.process(rgb)
            face_result=face_mesh.process(rgb)


            # Pose landmarks
            for i in range(25):
                            row[f"pose_{i}_x"] = 0
                            row[f"pose_{i}_y"] = 0
                            row[f"pose_{i}_z"] = 0
            if pose_result.pose_landmarks:
                mp_drawing.draw_landmarks(annotated_frame, pose_result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                for i, lm in enumerate(
                        pose_result.pose_landmarks.landmark[:25]):

                    row[f"pose_{i}_x"] = lm.x
                    row[f"pose_{i}_y"] = lm.y
                    row[f"pose_{i}_z"] = lm.z
            


            # Face landmarks
            if face_result.multi_face_landmarks:
                face_landmarks=face_result.multi_face_landmarks[0].landmark
                mp_drawing.draw_landmarks(annotated_frame, face_result.multi_face_landmarks[0], mp_face_mesh.FACEMESH_CONTOURS)
                for i, idx in enumerate(FACE_INDICES):
                    lm=face_landmarks[idx]
                    row[f"face_{i}_x"]=lm.x
                    row[f"face_{i}_y"]=lm.y
                    row[f"face_{i}_z"]=lm.z
                row["missing_face"]=False
            else:
                for i in range(70):
                    row[f"face_{i}_x"]=0
                    row[f"face_{i}_y"]=0
                    row[f"face_{i}_z"]=0
                row["missing_face"]=True
            # Hand landmarks
            for hand_idx in range(2):
                    for landmark_idx in range(21):
                        row[f"hand_{hand_idx}_{landmark_idx}_x"]=0
                        row[f"hand_{hand_idx}_{landmark_idx}_y"]=0
                        row[f"hand_{hand_idx}_{landmark_idx}_z"]=0                      
                         
            if hand_result.multi_hand_landmarks:

                for hand_idx, hand in enumerate(
                        hand_result.multi_hand_landmarks[:2]):
                    mp_drawing.draw_landmarks(annotated_frame, hand, mp_hands.HAND_CONNECTIONS)

                    for landmark_idx, lm in enumerate(hand.landmark):

                        row[f"hand_{hand_idx}_{landmark_idx}_x"] = lm.x
                        row[f"hand_{hand_idx}_{landmark_idx}_y"] = lm.y
                        row[f"hand_{hand_idx}_{landmark_idx}_z"] = lm.z


            row["missing_hand"] = (
                hand_result.multi_hand_landmarks is None
            )
            row["missing_pose"] = (
                pose_result.pose_landmarks is None
            )
            gif_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            gif_frame=cv2.resize(gif_frame, (640,360))
            gif_frames.append(Image.fromarray(gif_frame))
            rows.append(row)
            frame_number += 1

    gif_path=os.path.join(tempfile.gettempdir(), "skeleton_output.gif")
    original_gif_path=os.path.join(tempfile.gettempdir(),"uploaded_video.gif")
    if original_frames:
        original_frames[0].save(original_gif_path, save_all=True, append_images=original_frames[1:], duration=33, loop=0)

    if gif_frames:
        gif_frames[0].save(gif_path, save_all=True, append_images=gif_frames[1:], duration=33, loop=0)

    cap.release()

    return pd.DataFrame(rows), original_gif_path, gif_path


class SkeletonDMLRepresentation:
    def transform(self, x, y, z):
        n=3
        width=x.shape[1]
        if width % n !=0:
            extra_cols=width % n
            x=x[:,: width-extra_cols]
            y=y[:,: width-extra_cols]

        x =  np.reshape(x, (x.shape[0],-1,n))
        y =  np.reshape(y, (y.shape[0],-1,n))
        image=np.concatenate([x,y],axis=1)
        return image

def GCNtransform(df):
    T=min(len(df),80)
    skeleton = np.zeros((80,67,3))
    for t in range(T):
        row=df.iloc[t]
        for p in range(25):
            skeleton[t,p,0] = row[f"pose_{p}_x"]
            skeleton[t,p,1] = row[f"pose_{p}_y"]
            skeleton[t,p,2] = row[f"pose_{p}_z"]
        for h in range(21):
            skeleton[t, 25 + h, 0] = row[f"hand_0_{h}_x"]
            skeleton[t, 25 + h, 1] = row[f"hand_0_{h}_y"]
            skeleton[t, 25 + h, 2] = row[f"hand_0_{h}_z"]
        for h in range(21):
            skeleton[t, 46 + h, 0] = row[f"hand_1_{h}_x"]
            skeleton[t, 46 + h, 1] = row[f"hand_1_{h}_y"]
            skeleton[t, 46 + h, 2] = row[f"hand_1_{h}_z"]
    data=np.transpose(skeleton,(2,0,1))
    data=np.reshape(data,(3,80,67,1))
    data=np.expand_dims(data, axis=0)
    data= torch.tensor(data, dtype=torch.float32)
    print(data.shape)
    return data


if input_method=="Upload Video":
    uploaded_file=st.file_uploader("Upload a video", type=["mp4","avi","mov"])
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")as tmp:
            tmp.write(uploaded_file.read())
            st.session_state.video_path=tmp.name
elif input_method=="Record Using Webcam":
    st.subheader("Webcam Recorder")
    ctx=webrtc_streamer(key="recorder", video_processor_factory=VideoRecorder, media_stream_constraints={"video":True, "audio":False })
    st.info("Click 'Start' to activate the Webcam. Perform one sign for 3-5 seconds, then click 'Check Webcam Recording' before classifying. Clicking 'Stop' turns the Webcam off and deletes your sign.")
    # st.write("Webcam active:", ctx.state.playing)
    # if ctx.video_processor: 
    #     st.write("Frames captured:", len(ctx.video_processor.frames))
    # else: 
    #     st.write("Frames captured: 0")
    if ctx.state.playing:
        if st.button("Check Webcam Recording"):
             with st.spinner("Producing video..."):
                if ctx.video_processor is None:
                    st.error("Webcam not started")
                else:
                    frames=ctx.video_processor.frames
                    # st.write(f"Captured {len(frames)} frames")
                if len(frames)==0:
                    st.error("No webcam frames captured")
                else:
                    video_path=os.path.join(tempfile.gettempdir(), "webcam_sign.mp4")
                    height, width = frames[0].shape[:2]
                    writer=cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), 20, (width, height))
                    for frame in frames:
                        writer.write(frame)
                    writer.release()
                
                    ctx.video_processor.frames.clear()
                    st.session_state.video_path=video_path
                    preview_df, preview_original_gif, preview_skeleton_gif=(extract_video_data_from_path(video_path))
                    st.session_state.preview_df=preview_df
                    st.session_state.preview_original_gif=preview_original_gif
                    st.session_state.preview_skeleton_gif=preview_skeleton_gif
                    st.session_state.preview_ready=True
                    if st.session_state.get("preview_ready", False):
                        st.subheader("Reveiw Recording")
                        col1,col2=st.columns(2)
                        with col1:
                            st.write("Recorded Video")
                            st.image(st.session_state.preview_original_gif)
                        with col2:
                            st.write("Are you happy with this recording?")
                            st.info("If you are click 'Classify Sign'.")
                            st.write("Want to try again?")
                            st.info("Click 'Re-record' and try again.")
                            if st.button("Re-Record"):
                                if ctx.video_processor:
                                    ctx.video_processor.frames.clear()
                                st.session_state.preview_ready=False
                                st.session_state.video_path=None
                            # st.success("Recording discarded. Please sign again.")
                    st.success("Webcam recording saved")
                    st.info("Click 'Classify Sign' to run recognition.")


#Predict sign





def SLRpredict_video(df,SLRmodel):
    if df.empty:
        return "No sign detected", 0.0, torch.zeros(12)
    landmark_cols=[
        c for c in df.columns
        if c.endswith("_x")
        or c.endswith("_y")
        or c.endswith("_z")]
    excluded_body_landmarks=[10,11,13,14,19,20,21,22,23,24]
    excluded =tuple(f"pose_{i}" for i in excluded_body_landmarks)
    landmark_cols=[
        c for c in landmark_cols
        if not c.startswith(excluded)]
    x_cols=[c for c in landmark_cols if c.endswith("_x")]
    y_cols=[c for c in landmark_cols if c.endswith("_y")]
    z_cols=[c for c in landmark_cols if c.endswith("_z")]
    x = df[x_cols].to_numpy().T
    y = df[y_cols].to_numpy().T
    z = df[z_cols].to_numpy().T

    x = np.clip(x,0,1)
    y = np.clip(y,0,1)
    z = np.clip(z,0,1)
    image_method=SkeletonDMLRepresentation()
    image=image_method.transform(x,y,z)
    image=Image.fromarray(np.uint8(image*255)).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor()
    ])
    image=transform(image)
    image=image.float()
    image=image.unsqueeze(0)

    with torch.no_grad():
        output=SLRmodel(image)
        SLRprobs=torch.softmax(output,dim=1)
        pred=torch.argmax(SLRprobs, dim=1)
        SLRconfidence=SLRprobs.max().item()
    SLRsign=labels[pred.item()]
    return SLRsign, SLRconfidence, SLRprobs.squeeze()

def GCNpredict_video(df, GCNmodel):
    data=GCNtransform(df)
    with torch.no_grad():
        output=GCNmodel(data)
        GCNprobs=torch.softmax(output,dim=1)
        pred=torch.argmax(GCNprobs, dim=1)
    GCNsign=labels[pred.item()]
    GCNconfidence=GCNprobs.max().item()
    return GCNsign, GCNconfidence, GCNprobs.squeeze()


feedback_file="feedback.csv"
current_video=(st.session_state.video_path)
if current_video is not None:
    if st.button("Classify Sign"):
        st.session_state.classified=True
        st.session_state.feedback_submitted=False
        with st.spinner("Analysing video..."):
            df, original_gif_path, gif_path=extract_video_data_from_path(current_video)
            SLRsign, SLRconfidence, SLRprobs=SLRpredict_video(df, SLRmodel)
            GCNsign, GCNconfidence, GCNprobs=GCNpredict_video(df, GCNmodel)
        col1,col2=st.columns(2)
        with col1:
            title=("Webcam Recording"
                    if input_method=="Record Using Webcam"
                    else "Uploaded Video")
            st.subheader(title)
            st.image(original_gif_path)
        with col2:
            st.subheader("MediaPipe Skeleton")
            st.image(gif_path)
        col1,col2=st.columns(2)
        with col1:
            st.subheader("MobileNetV4")
            st.success(f"Predicted Sign: {SLRsign}")
            st.metric("Confidence",f"{SLRconfidence:.2%}")
            st.progress(float(SLRconfidence))
            if SLRprobs is not None:
                top3_prob, top3_idx = torch.topk(SLRprobs, 3)
                st.subheader("Top 3 Predictions")
                for p, idx in zip(top3_prob, top3_idx):
                    st.write(f"{labels[idx.item()]}:{p.item():.2%}")
        with col2:
            st.subheader("TD-GCN")
            st.success(f"Predicted Sign: {GCNsign}")
            st.metric("Confidence",f"{GCNconfidence:.2%}")
            st.progress(float(GCNconfidence))
            if GCNprobs is not None:
                top3_prob, top3_idx = torch.topk(GCNprobs, 3)
                st.subheader("Top 3 Predictions")
                for p, idx in zip(top3_prob, top3_idx):
                    st.write(f"{labels[idx.item()]}:{p.item():.2%}")
        if SLRsign == GCNsign:
            st.success(f"Both models agree: {SLRsign}")
        else:
            st.warning(f"Model disagree: " f"{SLRsign} vs {GCNsign}")
        comparison = pd.DataFrame({
            "Model": [
                "MobileNetV4",
                "TD-GCN"
            ],
            "Prediction": [
                SLRsign,
                GCNsign
            ],
            "Confidence": [
                f"{SLRconfidence:.2%}",
                f"{GCNconfidence:.2%}"
            ]
        })

        st.dataframe(comparison)
        st.session_state.SLRsign=SLRsign
        st.session_state.GCNsign=GCNsign
        st.session_state.SLRconfidence=SLRconfidence
        st.session_state.GCNconfidence=GCNconfidence
        st.session_state.df=df
        st.session_state.original_gif_path=original_gif_path
        st.session_state.gif_path=gif_path
        st.session_state.classified=True
        st.session_state.comparison=comparison
feedback_root=r"C:\Users\joann\OneDrive - Robert Gordon University\Project\Feedback"
os.makedirs(feedback_root, exist_ok=True)
if st.session_state.classified:
    # st.subheader("Last Prediction")
    # st.write(f"MobileNetV4: {st.session_state.SLRsign}")
    # st.write(f"TD-GCN: {st.session_state.GCNsign}")
    # st.dataframe(st.session_state.comparison)
    st.subheader("Help Improve Future Versions")
    st.write("Was the sign recognised correctly?")
    save_landmarks=st.checkbox("Allow anonymised landmark data to be saved for research")
    save_video=st.checkbox("Allow my original video recording to be saved")
    feedback=st.radio("Was the prediction correct?",
                      ["✅ Correct", "❌ Incorrect"])
    if feedback == "❌ Incorrect":
        correct_sign=st.selectbox("select the correct sign:", labels)
        if st.button("Submit Incorrect Feedback"):
            feedback_row=pd.DataFrame({
                "predicted_SLR_sign":[st.session_state.SLRsign],
                "predicted_GCN_sign":[st.session_state.GCNsign],
                "SLR_confidence": [st.session_state.SLRconfidence],
                "GCN_confidence": [st.session_state.GCNconfidence],
                "correct_label":[correct_sign],
                "feedback_type": ["corrected"],
                "timestamp":[pd.Timestamp.now()]
                })
            feedback_file=os.path.join(feedback_root, "feedback.csv")
            feedback_row.to_csv(feedback_file, mode="a", header=not os.path.exists(feedback_file), index=False)
            feedback_folder=os.path.join(feedback_root,"feedback_videos")
            os.makedirs(feedback_folder, exist_ok=True)
            new_name=f"{correct_sign}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            new_path=os.path.join(feedback_folder, new_name)
            if save_video:
                shutil.copy(st.session_state.video_path, new_path)
            if save_landmarks:
                landmark_folder=os.path.join(feedback_root, "feedback_landmarks")
                os.makedirs(landmark_folder, exist_ok=True)
                csv_name=(
                    f"{correct_sign}_"
                    f"{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
                    )
                landmark_path=os.path.join(landmark_folder,csv_name)
                st.session_state.df.to_csv(landmark_path, index=False)
            st.success("Thank you Your feedback has been recorded.")
    if feedback == "✅ Correct":
         if st.button("Submit Correct Feedback"):
            feedback_row=pd.DataFrame({
                "predicted_SLR_sign":[st.session_state.SLRsign],
                "predicted_GCN_sign":[st.session_state.GCNsign],
                "SLR_confidence": [st.session_state.SLRconfidence],
                "GCN_confidence": [st.session_state.GCNconfidence],
                "correct_label":[st.session_state.SLRsign],
                "feedback_type": ["confirmed"],
                "timestamp":[pd.Timestamp.now()]
                })
            feedback_file=os.path.join(feedback_root, "feedback.csv")
            feedback_row.to_csv(feedback_file, mode="a", header=not os.path.exists(feedback_file), index=False)
            feedback_folder=os.path.join(feedback_root,"feedback_videos")
            os.makedirs(feedback_folder, exist_ok=True)
            new_name=f"{st.session_state.SLRsign}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            new_path=os.path.join(feedback_folder, new_name)
            if save_video:
                shutil.copy(st.session_state.video_path, new_path)
            if save_landmarks:
                landmark_folder=os.path.join(feedback_root, "feedback_landmarks")
                os.makedirs(landmark_folder, exist_ok=True)
                csv_name=(
                    f"{st.session_state.SLRsign}_"
                    f"{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
                    )
                landmark_path=os.path.join(landmark_folder,csv_name)
                st.session_state.df.to_csv(landmark_path, index=False)
            st.session_state.feedback_submitted=True
            st.session_state.classified=False
            if st.session_state.feedback_submitted:
                st.success("Thank you Your feedback has been recorded.")
                st.stop()
if st.button("Reset Classification"):
    with st.spinner("Resetting..."):
        st.session_state.classified=False
        st.session_state.feedback_submitted=False
        for key in [
            "SLRsign", "GCNsign", "SLRconfidence", "GCNconfidence", "df", "comparison", "original_gif_path", "gif_path"]:
            if "video_path" in st.session_state:
                del st.session_state.video_path
            if key in st.session_state:
                del st.session_state[key]
            st.rerun()

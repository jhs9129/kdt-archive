import cv2
import numpy as np
import mediapipe as mp
import json
from collections import OrderedDict
import boto3
from botocore.exceptions import NoCredentialsError
import mysql.connector  
import os
from datetime import datetime, timedelta

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
print("1단계")
print("MediaPipe version:", mp.__version__)

def download_video_from_s3(bucket_name, video_name, local_path):
    s3 = boto3.client('s3', aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"), aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"))

    try:
        s3.download_file(bucket_name, f"{video_name}.mp4", local_path)
        print(f"Video downloaded from S3 to {local_path}")
    except NoCredentialsError:
        print("Credentials not available")
    
    print("download_video_from_s3 끝")
print(1)
def insert_data_to_rds(json_file_path):
    # RDS 연결 설정
    connection = mysql.connector.connect(
        host=os.environ.get("USER_HOST"),
        user=os.environ.get("USER_ID"),
        password=os.environ.get("USER_PASSWORD"),
        database=os.environ.get("USER_DB")
    )
    print("RDS 환경변수 설정 끝")
    cursor = connection.cursor()

    # JSON 파일 읽기
    with open(json_file_path, 'r') as json_file:
        data = json.load(json_file)

    # RDS 테이블에 데이터 삽입
    insert_query = "INSERT INTO USER_FM (tot_frame, detect_frame, proba, no_proba, name) VALUES (%s, %s, %s, %s, %s)"
    insert_data = (data["Total_Frame"], data["Detection_Frame"], data["Probability"], data["No_Probability"], data["Video_Name"])

    cursor.execute(insert_query, insert_data)
    connection.commit()

    # 연결 닫기
    cursor.close()
    connection.close()

    print("insert_data_to_rds 끝")
print(2)
def process_video(video_name):
    # S3에서 동영상 다운로드
    bucket_name = 'big4-team3'
    local_video_path = f"./{video_name}.mp4"
    download_video_from_s3(bucket_name, video_name, local_video_path)

    # 동영상 파일 열기
    cap = cv2.VideoCapture(local_video_path)

    # 총 프레임 수 
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 동영상이 정상적으로 열렸는지 확인
    if not cap.isOpened():
        print(f"비디오 파일을 찾을 수 없습니다: {local_video_path}")
        return

    print("데이터 추출 전")   # 여기까지 무조건 됨
    # mediapipe FaceMesh 인스턴스 생성
    with mp_face_mesh.FaceMesh(static_image_mode=False, 
                               max_num_faces=1,    
                               refine_landmarks=True,   
                               min_detection_confidence=0.5,
                               min_tracking_confidence=0.85) as face_mesh: 
        frame_count_with_landmarks = 0
        print("데이터 추출 시작")
        # 동영상 프레임을 순회하며 처리
        while cap.isOpened():
            # 동영상에서 프레임 읽기
            ret, frame = cap.read()
            if not ret:
                break
                
            # BGR을 RGB로 변환
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 얼굴 랜드마크 검출
            results = face_mesh.process(rgb_frame)
            # print("얼굴 검출 중")

            # 얼굴 랜드마크를 프레임에 표시
            if results.multi_face_landmarks:
                frame_count_with_landmarks += 1

                for face_landmarks in results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(image=frame, 
                                              landmark_list=face_landmarks, 
                                              connections=mp_face_mesh.FACEMESH_TESSELATION,
                                              landmark_drawing_spec=drawing_spec, 
                                              connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
                    mp_drawing.draw_landmarks(image=frame,
                                              landmark_list=face_landmarks,
                                              connections=mp_face_mesh.FACEMESH_CONTOURS,
                                              landmark_drawing_spec=drawing_spec,
                                              connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
                    mp_drawing.draw_landmarks(image=frame,
                                              landmark_list=face_landmarks,
                                              connections=mp_face_mesh.FACEMESH_IRISES,
                                              landmark_drawing_spec=None,
                                              connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style())

        # 처리가 끝나면 리소스 해제
        cap.release()
        print("데이터 추출 완료")

    Detection = frame_count_with_landmarks / total_frames
    N_Detection = 1 - Detection

    # 결과 데이터 추가
    data = {
        "Total_Frame": total_frames,
        "Detection_Frame": frame_count_with_landmarks,
        "Probability": Detection,
        "No_Probability": N_Detection,
        "Video_Name": video_name
    }

    # 결과 데이터 저장
    with open(f"{video_name}.json", 'w') as json_file:
        json.dump(data, json_file, indent=4)
        print('json파일 생성')
    try:
        # RDS에 데이터 삽입
        print("rds 데이터 삽입 전")
        insert_data_to_rds(f"{video_name}.json")
    except Exception as e:
        print("error occurred: ", str(e))
    # 끝
    print("데이터 삽입이 완료되었습니다.")
print(3)
def get_second_latest_video_name(bucket_name):
    s3 = boto3.client('s3', aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"), aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"))

    # S3 버킷 내에서 객체 목록을 가져옴
    response = s3.list_objects(Bucket=bucket_name)

    # 가장 최근에 업로드된 객체 선택
    second_latest_object = sorted(response['Contents'], key=lambda x: x['LastModified'], reverse=True)[1]

    # 파일 이름 반환 (확장자 제외)
    second_latest_video_name = os.path.splitext(os.path.basename(second_latest_object['Key']))[0]

    return second_latest_video_name

print(4)
if __name__ == '__main__':
    # S3 버킷 이름
    print("os.environ: ", os.environ)
    s3_bucket_name = 'big4-team3'

    # 최근에 업로드된 동영상 이름 가져오기
    latest_video_name = get_second_latest_video_name(s3_bucket_name)
    print('latest_video_name : ', latest_video_name)
    # 동영상 처리 함수 호출
    process_video(latest_video_name)
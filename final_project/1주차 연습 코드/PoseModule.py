import cv2
import mediapipe as mp
import time

class PoseDetector():

    def __init__(self,
                mode=False,
                model=1,
                smooth=True,
                enable_s=False,
                smooth_s=True,
                detectionCon=0.5,
                trackCon=0.5):
        # 포즈 디텍터를 지정된 매개변수와 함께 초기화합니다.
        self.mode = mode
        self.model = model
        self.smooth = smooth
        self.enable_s = enable_s
        self.smooth_s = smooth_s
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        # Mediapipe Pose 모듈 설정
        self.mpDraw = mp.solutions.drawing_utils # 랜드마크 및 감지된 객체 주위에 연결선 또는 경계 상자 그리는데 사용
        self.mpPose = mp.solutions.pose # 모델을 사용하기 위한 모듈
        self.pose = self.mpPose.Pose(self.mode, self.upBody, self.smooth,
                                     self.detectionCon, self.trackCon)
        
    def findPose(self, img, draw=True):
        # 이미지를 RGB 형식으로 변환합니다.
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 포즈 랜드마크를 찾기 위해 이미지를 처리합니다.
        results = self.pose.process(imgRGB)
        
        # 랜드마크와 연결선을 이미지에 그립니다 (선택적).
        if results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, results.pose_landmarks,
                                            self.mpPose.POSE_CONNECTIONS)
        return img      

def main():
    # 지정된 비디오 파일에 대한 비디오 캡처 객체를 엽니다.
    cap = cv2.VideoCapture('./1.mp4')  # 비디오 파일의 경로를 수정하세요.
    pTime = 0
    detector = PoseDetector()  # PoseDetector 클래스의 인스턴스를 생성합니다.
    while True:
        # 비디오에서 한 프레임을 읽어옵니다.
        success, img = cap.read()
        
        # 프레임에서 포즈 랜드마크를 찾아 그립니다.
        img = detector.findPose(img)

        # 초당 프레임 수 (fps)를 계산하고 표시합니다.
        cTime = time.time() 
        fps = 1 / (cTime - pTime)
        pTime = cTime

        # 이미지에 fps를 표시합니다.
        cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, 
                    (255, 0, 0), 3)
        
        # 랜드마크가 있는 이미지를 표시합니다.
        cv2.imshow("Image", img)

        # 'ESC' 키가 눌리면 루프를 종료합니다.
        if cv2.waitKey(1) == 27:
            break

    # 비디오 캡처 객체를 해제하고 모든 창을 닫습니다.
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

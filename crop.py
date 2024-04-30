import cv2, os

name = "가렵다"

# 파일 경로명 주의
video_files = ["C:/Users/_/Desktop/video/ID_9/가렵다(측).avi", "C:/Users/_/Desktop/video/ID_9/기절(측).avi",
               "C:/Users/_/Desktop/video/ID_9/부러지다(측).avi",]

# 인덱스 0(가렵다), 1(기절), 2(부러지다), 3(어제), 4(어지러움), 5(열나다), 6(오늘), 7(진통제), 8(창백하다), 9(토하다)
output_folder = "crop"
os.makedirs(output_folder, exist_ok=True)

for video_file in video_files:
    cap = cv2.VideoCapture(video_file)

    output_path = os.path.join(output_folder, os.path.basename(f"{os.path.basename(video_file)}"))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')    # .avi 확장자로 설정
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (1080, 800))   # 30.0은 프레임

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 픽셀 크기 조절
        cropped_frame = frame[550:1350, :]
            
        # 조절된 프레임을 새로운 동영상에 추가
        out.write(cropped_frame)

        # 영상을 화면에 표시
        cv2.imshow('MediaPipe', cropped_frame)
        if cv2.waitKey(1) == ord('q'):
            break
    
    # 작업 완료 후 해제
    cap.release()
    out.release()

print("영상 크기 조절 및 저장 완료")
# 가렵다(정) 2.03MB -> 134KB
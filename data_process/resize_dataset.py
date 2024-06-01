import cv2, os

name = "토하다"

# 파일 경로명 주의
video_files = [f"C:/Users/mshof/Desktop/video/ID_1/{name}(정).avi", f"C:/Users/mshof/Desktop/video/ID_1/{name}(측).avi",
               f"C:/Users/mshof/Desktop/video/ID_2/{name}(정).avi", f"C:/Users/mshof/Desktop/video/ID_2/{name}(측).avi",
               f"C:/Users/mshof/Desktop/video/ID_3/{name}(정).avi", f"C:/Users/mshof/Desktop/video/ID_3/{name}(측).avi",
               f"C:/Users/mshof/Desktop/video/ID_4/{name}(정).avi", f"C:/Users/mshof/Desktop/video/ID_4/{name}(측).avi",
               f"C:/Users/mshof/Desktop/video/ID_5/{name}(정).avi", f"C:/Users/mshof/Desktop/video/ID_5/{name}(측).avi",
               f"C:/Users/mshof/Desktop/video/ID_6/{name}(정).avi", f"C:/Users/mshof/Desktop/video/ID_6/{name}(측).avi",
               f"C:/Users/mshof/Desktop/video/ID_7/{name}(정).avi", f"C:/Users/mshof/Desktop/video/ID_7/{name}(측).avi",
               f"C:/Users/mshof/Desktop/video/ID_8/{name}(정).avi", f"C:/Users/mshof/Desktop/video/ID_8/{name}(측).avi",
               f"C:/Users/mshof/Desktop/video/ID_9/{name}(정).avi", f"C:/Users/mshof/Desktop/video/ID_9/{name}(측).avi",
               f"C:/Users/mshof/Desktop/video/ID_10/{name}(정).avi", f"C:/Users/mshof/Desktop/video/ID_10/{name}(측).avi",]

# 인덱스 0(가렵다), 1(기절), 2(부러지다), 3(어제), 4(어지러움), 5(열나다), 6(오늘), 7(진통제), 8(창백하다), 9(토하다)
output_folder = "C:/Users/mshof/Desktop/resized_video/resized_video_0"
os.makedirs(output_folder, exist_ok=True)

# 너비, 높이 설정
# 원본 영상과 비율은 동일하게 유지. 1280x720 (1.78 : 1)
new_width = 455
new_height = 256

# 픽셀 크기 설정
distance_from_center = 200

center_x = new_width // 2
center_y = new_height // 2

left_x = center_x - distance_from_center
right_x = center_x + distance_from_center

idx = 1

for video_file in video_files:
    cap = cv2.VideoCapture(video_file)

    output_path = os.path.join(output_folder, os.path.basename(f"{idx}_{os.path.basename(video_file)}"))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')    # .avi 확장자로 설정
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (right_x-left_x, new_height))   # 30.0은 프레임

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 해상도 조절
        resized_frame = cv2.resize(frame, (new_width, new_height))

        # 픽셀 크기 조절
        cropped_frame = resized_frame[:, left_x:right_x]
            
        # 조절된 프레임을 새로운 동영상에 추가
        out.write(cropped_frame)

        # 영상을 화면에 표시
        cv2.imshow('MediaPipe', cropped_frame)
        if cv2.waitKey(1) == ord('q'):
            break

    idx += 1
    
    # 작업 완료 후 해제
    cap.release()
    out.release()

print("영상 크기 조절 및 저장 완료")
# 가렵다(정) 2.03MB -> 116KB
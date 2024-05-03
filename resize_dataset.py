import cv2, os

name = ""

# 파일 경로명 주의
video_files = [f"C:/Users/mshof/Desktop/video/test_video/가렵다.mp4", f"C:/Users/mshof/Desktop/video/test_video/기절.mp4",
               f"C:/Users/mshof/Desktop/video/test_video/부러지다.mp4", f"C:/Users/mshof/Desktop/video/test_video/어제.mp4",
               f"C:/Users/mshof/Desktop/video/test_video/어지러움.mp4", f"C:/Users/mshof/Desktop/video/test_video/열나다.mp4",
               f"C:/Users/mshof/Desktop/video/test_video/오늘.mp4", f"C:/Users/mshof/Desktop/video/test_video/진통제.mp4",
               f"C:/Users/mshof/Desktop/video/test_video/창백하다.mp4", f"C:/Users/mshof/Desktop/video/test_video/토하다.mp4"]

# 인덱스 0(가렵다), 1(기절), 2(부러지다), 3(어제), 4(어지러움), 5(열나다), 6(오늘), 7(진통제), 8(창백하다), 9(토하다)
output_folder = "C:/Users/mshof/Desktop/video/test_resize"
os.makedirs(output_folder, exist_ok=True)

# 너비, 높이 설정
# 원본 영상과 비율은 동일하게 유지. 1280x720 (1.78 : 1)
new_width = 400
new_height = 300

# 픽셀 크기 설정
distance_from_center = 300

center_x = new_width // 2
center_y = new_height // 2

left_x = center_x - distance_from_center
right_x = center_x + distance_from_center

idx = 1

for video_file in video_files:
    cap = cv2.VideoCapture(video_file)

    output_path = os.path.join(output_folder, os.path.basename(f"{idx}_{os.path.basename(video_file)}"))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')    # .avi 확장자로 설정
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (new_width, new_height))   # 30.0은 프레임

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 프레임 크기 조절
        resized_frame = cv2.resize(frame, (new_width, new_height))

        # 픽셀 크기 조절
        # cropped_frame = resized_frame[:, left_x:right_x]
            
        # 조절된 프레임을 새로운 동영상에 추가
        out.write(resized_frame)

        # 영상을 화면에 표시
        cv2.imshow('MediaPipe', resized_frame)
        if cv2.waitKey(1) == ord('q'):
            break

    idx += 1
    
    # 작업 완료 후 해제
    cap.release()
    out.release()

print("영상 크기 조절 및 저장 완료")
# 가렵다(정) 2.03MB -> 134KB
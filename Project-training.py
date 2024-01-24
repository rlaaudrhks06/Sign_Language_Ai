import cv2
import mediapipe as mp
import numpy as np
import keyboard
import time
import pandas as plt

max_num_hands = 1 #최대 감지할 손수

gesture = {
    0:'a',1:'b',2:'c',3:'d',4:'e',5:'f',6:'g',7:'h',
    8:'i',9:'j',10:'k',11:'l',12:'m',13:'n',14:'o',
    15:'p',16:'q',17:'r',18:'s',19:'t',20:'u',21:'v',
    22:'w',23:'x',24:'y',25:'z',26:'spacing',27:'clear',28:'FUCk'
}#학습할 제스처

mp_hands = mp.solutions.hands #연두색으로 손가락 마디 표시
mp_drawing = mp.solutions.drawing_utils #점으로 손가락 마디 표시
hands = mp_hands.Hands(
    max_num_hands = max_num_hands,
    min_detection_confidence = 0.5,
    min_tracking_confidence = 0.5
)
# mp_hands.Hands은 손가락을 이식하는 모듈을 초기화
# min_detection_confidence은 탐지가 성공한 것으로 간주하는 사람 탐지 모델의 최소 신뢰값
# min_tracking_confidence은 추적에 실패하면 다음 이미지 입력에서 사람 감지가 자동으로 호출됩니다.
# https://puleugo.tistory.com/17


f = open('test.txt', 'w')

file = np.genfromtxt('python/Sign_Language/dataSet.txt', delimiter=',')
angleFile = file[:,:-1]
labelFile = file[:,-1]
angle = angleFile.astype(np.float32)
lable = labelFile.astype(np.float32)

#OpenCV k 최근접 이웃 알고리즘
#cv2.ROW_SAMPLE : 하나의 데이터가 한 행으로 구성
knn = cv2.ml.KNearest_create() # K최접근 알고리즘중에서 OPENCV과 같이 사용할 수 있는 알고리즘
knn.train(angle,cv2.ml.ROW_SAMPLE,lable) # ANGLE, CV.2ML.ROW_SAMPLE, LABEL 을 학습 시킨다
cap = cv2.VideoCapture(0)

startTime = time.time()
prev_index = 0
sentence = ''
recognizeDelay = 1 
while True:
    ret,img = cap.read() # 재생되는 비디오의 한 프레임을 읽는다.
    frame = cv2.flip(img, 1)
    if not ret:
        continue

    imgRGB = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB) #프레임을 대각선으로 변환합니다
    result = hands.process(imgRGB)

    if result.multi_hand_landmarks is not None:
        for res in result.multi_hand_landmarks: 
            joint = np.zeros((21,3))
            for j,lm in enumerate(res.landmark):
                joint[j] = [lm.x,lm.y,lm.z]
            
            v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9,10,11, 0,13,14,15, 0,17,18,19],:]
            v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20],:]

            v = v2 - v1
            v = v / np.linalg.norm(v, axis=1)[:,np.newaxis]
            compareV1 = v[[0, 1, 2, 4, 5, 6, 7, 8, 9,10,12,13,14,16,17],:]
            compareV2 = v[[1, 2, 3, 5, 6, 7, 9,10,11,13, 14, 15, 17, 18, 19],:]
            angle = np.arccos(np.einsum('nt, nt->n', compareV1, compareV2))
            
            angle = np.degrees(angle)
            # if keyboard.is_pressed('a'): #a를 누를시 현재 데이터(angle)가 txt파일에 저장
            #     for num in angle:
            #         num = round(num,6)
            #         f.write(str(num))
            #         f.write(',')
            #     f.write("0.000000") #데이터를 저장할 gesture의 label 번호 27로
            #     f.write('\n')  
                # print("next")
            data = np.array([angle], dtype=np.float32)
            ret, results, neighbours, dist = knn.findNearest(data,3)
            index= int(results[0][0])
            if index in gesture.keys():
                if index != prev_index:
                    startTime = time.time()
                    prev_index = index
                else:
                    if time.time() - startTime > recognizeDelay:
                        if index == 26:
                            sentence += ' '
                        elif index == 27:
                            sentence = ''
                        else:
                            sentence += gesture[index]
                        startTime = time.time()

                cv2.putText(frame, gesture[index].upper(),(int(res.landmark[0].x * img.shape[1] - 10), 
                            int(res.landmark[0].y * img.shape[0] + 40)), cv2.FONT_HERSHEY_SIMPLEX, 1,color=(255,255,255))
            
            mp_drawing.draw_landmarks(frame, res, mp_hands.HAND_CONNECTIONS)
    cv2.putText(frame, sentence, (20,440), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255),3)


    cv2.imshow('HandTracking', frame)
    cv2.waitKey(1)
    # if keyboard.is_pressed('b'):
    #     break
f.close()

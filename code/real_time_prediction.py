# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2
import hyperparameters as hp

def real_time_prediction(model):
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()
    time.sleep(2.0)
    fps = FPS().start()
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    counter = 0
    while True:
        frame = vs.read()
        # frame = imutils.resize(frame, width=400)
        (h, w) = frame.shape[:2]
        print("Frame h", h)
        print("Frame w", w)
        frame = imutils.resize(frame, width=400)
        npframe = np.array(frame, dtype=np.uint8)
        print("Np Frame shape", np.shape(npframe))
        # gray_image = rgb2gray(model_frame)
        gray_img = cv2.cvtColor(npframe, cv2.COLOR_BGR2GRAY)
        new_image = npframe
        filter_frame = frame


        faces = face_cascade.detectMultiScale(gray_img, 1.1, 4)

        if (len(faces) > 0):
            counter += 1
            (fx, fy, fw, fh) = faces[0]
            new_image = cv2.rectangle(
                new_image, (fx, fy), (fx + fw, fy + fh), (255, 0, 0), 2)

            bounding_box = gray_img[fx: fx + fw, fy: fy + fh]
            print("Bounding box shape", np.shape(bounding_box))
            width, height = np.shape(bounding_box)

            model_input = cv2.resize(
                bounding_box, (hp.image_dim, hp.image_dim)) / 255

            model_input = np.reshape(
                model_input, (1, hp.image_dim, hp.image_dim, 1))

            prds = model.predict(model_input) / 96

            ptxs = (prds[0][0::2] * width) + fx
            ptys = (prds[0][1::2] * height) + fy

            print("xs", ptxs)
            print("ys", ptys)

            new_image = cv2.circle(
                new_image, (int(ptxs[0]), int(ptys[0])), 2, (0, 0, 255))
            new_image = cv2.circle(
                new_image, (int(ptxs[1]), int(ptys[1])), 2, (0, 0, 255))
            new_image = cv2.circle(
                new_image, (int(ptxs[2]), int(ptys[2])), 2, (0, 0, 255))
            new_image = cv2.circle(
                new_image, (int(ptxs[3]), int(ptys[3])), 2, (0, 0, 255))
            new_image = cv2.circle(
                new_image, (int(ptxs[4]), int(ptys[4])), 2, (0, 0, 255))
            new_image = cv2.circle(
                new_image, (int(ptxs[5]), int(ptys[5])), 2, (0, 0, 255))
            new_image = cv2.circle(
                new_image, (int(ptxs[6]), int(ptys[6])), 2, (0, 0, 255))
            new_image = cv2.circle(
                new_image, (int(ptxs[7]), int(ptys[7])), 2, (0, 0, 255))
            new_image = cv2.circle(
                new_image, (int(ptxs[8]), int(ptys[8])), 2, (0, 0, 255))
            new_image = cv2.circle(
                new_image, (int(ptxs[9]), int(ptys[9])), 2, (0, 0, 255))
            new_image = cv2.circle(
                new_image, (int(ptxs[10]), int(ptys[10])), 2, (0, 0, 255))
            new_image = cv2.circle(
                new_image, (int(ptxs[11]), int(ptys[11])), 2, (0, 0, 255))
            new_image = cv2.circle(
                new_image, (int(ptxs[12]), int(ptys[12])), 2, (0, 0, 255))
            new_image = cv2.circle(
                new_image, (int(ptxs[13]), int(ptys[13])), 2, (0, 0, 255))
            new_image = cv2.circle(
                new_image, (int(ptxs[14]), int(ptys[14])), 2, (0, 0, 255))

            cool_glasses = cv2.imread('cool_guy_shades.png', -1)
            filter_frame = np.copy(frame)
            print("Filter frame shape", np.shape(filter_frame))
            cool_glasses_width = int(ptxs[hp.left_eye_outer_corner]) - int(ptxs[hp.right_eye_outer_corner])
            print("Glasses width", cool_glasses_width)
            cool_glasses = cv2.resize(cool_glasses, (2 * cool_glasses_width, 50))
            print("Shades Shape", np.shape(cool_glasses))
            cgw, cgh, cgd = np.shape(cool_glasses)
        
            for gx in range(cgw):
                for gy in range(cgh):
                    if(cool_glasses[gx][gy][3] != 0):
                        right_eye_start = int(ptxs[hp.right_eye_outer_corner])
                        brow_start = int(ptys[hp.right_eyebrow_outer_end])
                        print("left eye start", right_eye_start)
                        print("left eyebrow start", brow_start)
                        filter_frame[gx + brow_start - 10][gy + right_eye_start - 5] = cool_glasses[gx][gy][0]



            rainbow = cv2.imread('rainbow_mouth.png', -1)
            rainbow_height = 100
            rainbow_width = 100
            rainbow = cv2.resize(rainbow, (rainbow_width, rainbow_height)) 

            rw, rh, rd = np.shape(rainbow)

            for rx in range(rw):
                for ry in range(rh):
                    if(rainbow[rx][ry][3] != 0):
                        lip_x = int(ptxs[hp.mouth_center_top_lip])
                        lip_y = int(ptys[hp.mouth_center_top_lip])
                        if (counter % 2 == 0):
                            filter_frame[rx + lip_y - 2][ry + lip_x - 30][0] = rainbow[rx][ry][0]
                            filter_frame[rx + lip_y - 2][ry + lip_x - 30][1] = rainbow[rx][ry][1]
                            filter_frame[rx + lip_y - 2][ry + lip_x - 30][2] = rainbow[rx][ry][2]
                        else:
                            filter_frame[rx + lip_y - 2][ry + lip_x - 30][2] = rainbow[rx][ry][0]
                            filter_frame[rx + lip_y - 2][ry + lip_x - 30][1] = rainbow[rx][ry][1]
                            filter_frame[rx + lip_y - 2][ry + lip_x - 30][0] = rainbow[rx][ry][2]


        # print("Prediction pts shape", np.shape(prd))
            print("Getting image")
            print("Dimensions height", np.shape(frame))
        # circle = cv2.circle(new_image, (48,48), 3, (0, 0, 255))
        cv2.imshow("Frame", frame)
        cv2.imshow("SecondFrame", new_image)
        cv2.imshow("FilterFrame", filter_frame)
        key = cv2.waitKey(1) & 0xFF
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
        fps.update()
    fps.stop()
    cv2.destroyAllWindows()
    vs.stop()

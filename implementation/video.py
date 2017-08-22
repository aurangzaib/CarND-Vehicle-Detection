import glob

import cv2 as cv


class Video:
    @staticmethod
    def img2video(folder):
        img = cv.imread("../buffer/detections/1503358539-detection.png")
        dimensions = img.shape[1], img.shape[0]
        out = cv.VideoWriter('pipeline.avi',
                             cv.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                             10,
                             dimensions)
        imgs = glob.glob(folder)
        for filename in imgs:
            frame = cv.imread(filename)
            cv.imshow("image", frame)
            out.write(frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

        out.release()

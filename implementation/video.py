import glob

import cv2 as cv


class Video:
    @staticmethod
    def img2video(folder):
        img = cv.imread("../buffer/pipeline-original/binary-original-1502615350.jpg")
        dimensions = img.shape[1], img.shape[0]
        # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
        out = cv.VideoWriter('pipeline.avi',
                             cv.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                             10,
                             dimensions)
        imgs = glob.glob(folder)

        for filename in imgs:
            frame = cv.imread(filename)
            out.write(frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

        out.release()

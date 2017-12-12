import cv2
import numpy as np

stream1 = 'http://USERNAME:PASSWORD@IP:PORT/++video?cameraNum=2&74258'
stream2 = 'http://IP:PORT/video'

# stream1_bytes = ''
# stream2_bytes = ''

# tilt = np.pi * 35 / 120

# plt.axis([1, 2, 1, 2])
# plt.ion()
cap = cv2.VideoCapture(stream1)
cap2 = cv2.VideoCapture(stream2)

pts1 = np.float32([
        [0, 0],
        [800, 0],
        [800, 480],
        [0, 480]
    ])

pts2 = np.float32([
    [0, 0],
    [800, 0],
    [533, 480],
    [266, 480]
])
M = cv2.getPerspectiveTransform(pts1, pts2)

while True:
    ret, frame = cap.read()
    ret2, frame2 = cap2.read()
    # stream1_bytes += stream1.read(1024)
    # a1 = stream1_bytes.find('\xff\xd8')
    # b1 = stream1_bytes.find('\xff\xd9')

    # stream2_bytes += stream2.read(1024)
    # a2 = stream2_bytes.find('\xff\xd8')
    # b2 = stream2_bytes.find('\xff\xd9')

    # img1, img2, dst1, dst2 = [None] * 4

    # if a1 != -1 and b1 != -1:
    #     pts1 = np.float32([[0, 40], [300, 40], [0, 400], [300, 400]])
    #     pts2 = np.float32([[0, 0], [300, 0], [100, 400], [200, 400]])
    #     M = cv2.getPerspectiveTransform(pts1, pts2)
    #     jpg1 = stream1_bytes[a1:b1 + 2]
    #     stream1_bytes = stream1_bytes[b1 + 2:]
    #     img1 = cv2.imdecode(np.fromstring(jpg1, dtype=np.uint8), 1)
    #     dst1 = cv2.warpPerspective(img1, M, (300, 300))

    # if a2 != -1 and b2 != -1:
    #     jpg2 = stream2_bytes[a2:b2 + 2]
    #     stream1_bytes = stream2_bytes[b2 + 2:]
    #     img2 = cv2.imdecode(np.fromstring(jpg2, dtype=np.uint8), 1)
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    # [center_x - scale, center_y - scale * hwratio],  # top left
    # [center_x + scale, center_y - scale * hwratio],  # top right
    # [center_x + scale, center_y + scale * hwratio],  # bottom right
    # [center_x - scale, center_y + scale * hwratio],  # bottom left
    # 800 x 480

    dst1 = cv2.warpPerspective(frame, M, (800, 600))
    dst2 = cv2.warpPerspective(frame2, M, (800, 600))

    both = np.hstack((dst1, dst2))

    cv2.imshow('frame', both)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        # rows, cols, ch = img2.shape
        #
        # if not img1 is None and not img2 is None:
        #     plt.subplot(2,2,1), plt.imshow(img1), plt.title('Input HOME')
        #     plt.subplot(2,2,2), plt.imshow(dst1), plt.title('Output HOME')
        #
        #     plt.subplot(221), plt.imshow(img2), plt.title('Input Office')
        #     plt.subplot(222), plt.imshow(dst2), plt.title('Output Office')
        #
        # plt.pause(0.01)
        # plt.show()

        # plt.
        # plt.imshow('i',plt)
        # cv2.imshow('i',i)
        # if cv2.waitKey(1) ==27:
        #     exit(0)
cap.release()
cv2.destroyAllWindows()

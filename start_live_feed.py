import cv2
import numpy as np
import imutils
from keras.models import load_model
import matplotlib.pyplot as plt


def mean_squared_loss(x1, x2):
    ''' Compute Euclidean Distance Loss  between
    input frame and the reconstructed frame'''

    diff = x1-x2
    a, b, c, d, e = diff.shape
    n_samples = a*b*c*d*e
    sq_diff = diff**2
    Sum = sq_diff.sum()
    dist = np.sqrt(Sum)
    mean_dist = dist/n_samples

    return mean_dist


threshold = 0.00078
loss_values = []
frame_values = []
n = 0

test_location = r'C:\Users\Kiran\Desktop\Capstone\Anomalous_Event_Detection\Avenue_test\11.avi'
#test_location = r"C:\Users\Kiran\Desktop\Capstone\Anomalous_Event_Detection\Random_Testing_Video_Trim.mp4"
vc = cv2.VideoCapture(test_location)
rval = True

print('Loading model')
model = load_model('Anomaly_detector_Avenue.h5')
print('Model loaded')

while vc.isOpened():
    imagedump = []
    rval, frame = vc.read()
    n = n + 1
    if rval:
        for i in range(10):
            image = imutils.resize(frame, width=700, height=600)
            frame = cv2.resize(frame, (227, 227), interpolation=cv2.INTER_AREA)

            # Convert the Image to Grayscale
            image = frame
            gray = 0.2989*frame[:, :, 0]+0.5870 * \
                frame[:, :, 1]+0.1140*frame[:, :, 2]
            gray = (gray-gray.mean())/gray.std()
            gray = np.clip(gray, 0, 1)
            imagedump.append(gray)
    else:
        break
    imagedump = np.array(imagedump)
    imagedump.resize(227, 227, 10)
    imagedump = np.expand_dims(imagedump, axis=0)
    imagedump = np.expand_dims(imagedump, axis=4)

    output = model.predict(imagedump)

    loss = mean_squared_loss(imagedump, output)

    print('Loss:', loss)
    loss_values.append(loss)
    frame_values.append(n)

    if loss > threshold:
        print('Anomalies Detected')
        cv2.putText(image, "Abnormal Event", (50, 30),
                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (36, 13, 214))
    else:
        print("Normal Event")

    key = cv2.waitKey(20)
    if key == 27:
        break

    cv2.namedWindow("Live Feed", cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Live Feed', 750, 600)
    cv2.imshow("Live Feed", frame)

plt.plot(frame_values, loss_values)
plt.xlabel('Frames')
plt.ylabel('Loss')
plt.title('Frame - loss curve')
plt.show()


cv2.destroyWindow("Live Feed")
vc.release()

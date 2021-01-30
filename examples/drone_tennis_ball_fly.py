import sys
import traceback
import tellopy
import av
import cv2.cv2 as cv2  # for avoidance of pylint error
import numpy
import time
from collections import deque
#from imutils.video import VideoStream
import imutils

def main():
    drone = tellopy.Tello()
    buffer_len = 64
    try:
        drone.connect()
        drone.wait_for_connection(60.0)
        drone.takeoff()
        time.sleep(1)

        retry = 3
        container = None
        while container is None and 0 < retry:
            retry -= 1
            try:
                container = av.open(drone.get_video_stream())
            except av.AVError as ave:
                print(ave)
                print('retry...')

        frame_skip = 0
        greenLower = (29, 86, 6)
        greenUpper = (64, 255, 255)
        pts = deque(maxlen=buffer_len)
        targetRadius = 20
        pRadius = 1.5
        targetY = 261
        pY = 0.12
        targetX = 323
        pX = 0.1

        while True:
            for frame in container.decode(video=0):
                if 0 < frame_skip:
                    frame_skip = frame_skip - 1
                    continue
                start_time = time.time()
                image = cv2.cvtColor(numpy.array(frame.to_image()), cv2.COLOR_RGB2BGR)
                if image is None:
                   break
                image = imutils.resize(image, width=600)
                blurred = cv2.GaussianBlur(image, (11, 11), 0)
                hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv, greenLower, greenUpper)
                mask = cv2.erode(mask, None, iterations=2)
                mask = cv2.dilate(mask, None, iterations=2)
                cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE)
                cnts = imutils.grab_contours(cnts)

                center = None
                radius = 0
                y = 0
                x = 0

                if len(cnts) > 0:
                    c = max(cnts, key=cv2.contourArea)    
                    ((x, y), radius) = cv2.minEnclosingCircle(c)
                    M = cv2.moments(c)
                    center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

                # only proceed if the radius meets a minimum size
                    if radius > 10:
                # draw the circle and centroid on the frame,
                # then update the list of tracked points
                            cv2.circle(image, (int(x), int(y)), int(radius),
                                (0, 255, 255), 2)
                            cv2.circle(image, center, 5, (0, 0, 255), -1)           
                pts.appendleft(center)
                for i in range(1, len(pts)):
                    if pts[i - 1] is None or pts[i] is None:
                        continue
                    thickness = int(numpy.sqrt(buffer_len / float(i + 1)) * 2.5)
                    cv2.line(image, pts[i - 1], pts[i], (0, 0, 255), thickness)
                # show the frame to our screen
                cv2.imshow("Image", image)
                
                #move drone to target radius
                if radius > 15:
                    errorRadius = radius - targetRadius
                    distanceToFly = pRadius * errorRadius
                    if distanceToFly < 0:
                       drone.forward(-distanceToFly)
                    else:
                       drone.backward(distanceToFly)

                    errorY = y - targetY
                    YdistanceToFly = pY * errorY
                    if YdistanceToFly < 0:
                        drone.up(-YdistanceToFly)
                    else:
                        drone.down(YdistanceToFly)

                    errorX = x - targetX
                    XdistanceToFly = pX * errorX
                    if XdistanceToFly < 0:
                        drone.left(-XdistanceToFly)
                    else:
                        drone.right(XdistanceToFly)
    
            # if the 'q' key is pressed, stop the loop
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    # drone.down(50)
                    # time.sleep(5)
                    # drone.land()
                    # time.sleep(5)
                    raise
                if frame.time_base < 1.0/60:
                    time_base = 1.0/60
                else:
                    time_base = frame.time_base
                frame_skip = int((time.time() - start_time)/time_base)

    except Exception as ex:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback)
        print(ex)

    finally:
        drone.down(50)
        time.sleep(5)
        drone.land()
        time.sleep(5)
        drone.quit()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

    #100 miliseconds one frame
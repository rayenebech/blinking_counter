import dlib
import cv2
import numpy as np
from numpy.linalg import norm
import time
import RPi.GPIO as GPIO


import Adafruit_GPIO.SPI as SPI
import Adafruit_SSD1306

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont


FRAME_SIZE = (200,150)
BLUE = 17
RED =  18
GREEN = 27 
STOP_BUTTON = 22
START_BUTTON = 23



# Raspberry Pi pin configuration:
RST = None     # on the PiOLED this pin isnt used
# Note the following are only used with SPI:
DC = 23
SPI_PORT = 0
SPI_DEVICE = 0


class BlinkDetector():

    def __init__(self):
        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(BLUE, GPIO.OUT)
        GPIO.setup(RED, GPIO.OUT)
        GPIO.setup(GREEN, GPIO.OUT)

        GPIO.setup(STOP_BUTTON, GPIO.IN, pull_up_down = GPIO.PUD_UP) #Button
        GPIO.setup(START_BUTTON, GPIO.IN, pull_up_down = GPIO.PUD_UP) #Button

        self.blink_count = 0 
        self.start_time = time.time()
        self.sure = 60

        self.threshold = 0.25
        self.eye_closed = False
        self.close_all()
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        self.disp = Adafruit_SSD1306.SSD1306_128_32(rst=RST)


    def setup_display(self):
        self.disp.begin()

        # Clear display.
        self.disp.clear()
        self.disp.display()

        # Create blank image for drawing.
        # Make sure to create image with mode '1' for 1-bit color.
        self.width = self.disp.width
        self.height = self.disp.height
        self.image = Image.new('1', (self.width, self.height))

        # Get drawing object to draw on image.
        self.draw = ImageDraw.Draw(self.image)

        # Draw a black filled box to clear the image.
        self.draw.rectangle((0,0,self.width, self.height), outline=0, fill=0)

    def draw_text(self, count):
        # Draw some shapes.
        # First define some constants to allow easy resizing of shapes.
        finish_time = int(time.time() - self.start_time)
        padding = -2
        top = padding
        bottom = self.height-padding
        # Move left to right keeping track of the current x position for drawing shapes.
        x = 0
        # Load default font.
        font = ImageFont.load_default()
        self.draw.rectangle((0,0,self.width,self.height), outline=0, fill=0)
        self.draw.text((x, top),       "Counter: " + str(count),  font=font, fill=255)
        self.draw.text((x, top+16),       "Time (s): " + str(finish_time),  font=font, fill=255)
        self.disp.image(self.image)
        self.disp.display()
        time.sleep(0.001)
 
    def close_all(self):
        GPIO.output(BLUE, GPIO.LOW)
        GPIO.output(RED, GPIO.LOW)
        GPIO.output(GREEN, GPIO.LOW)


    def blink_led(self, count):
        self.close_all()
        if count < 9:
            GPIO.output(BLUE, GPIO.HIGH)
            return
        if count < 15:
            GPIO.output(GREEN, GPIO.HIGH)
            return
        GPIO.output(RED, GPIO.HIGH)
        return


    def run(self):
        print("SYSTEM IS READY!")
        while GPIO.input(START_BUTTON) == True:
            pass
        print("pressed button")
        self.setup_display()        
        self.close_all()
        self.start_time = time.time()
        vs = cv2.VideoCapture(0)
        print("camera is opened")
        while True:
            _, frame = vs.read()
            frame = cv2.resize(frame, FRAME_SIZE)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects = self.detector(gray, 0)

            for rect in rects:
                landmarks = self.predictor(gray, rect)

                # Use the coordinates of each eye to compute the eye aspect ratio.
                right_aspect_ratio = self.aspect_ratio(landmarks, range(36, 42))
                left_aspect_ratio = self.aspect_ratio(landmarks, range(42, 48))
                ear = (left_aspect_ratio + right_aspect_ratio) / 2.0

                if ear < self.threshold:
                    self.eye_closed = True

                if ear >= self.threshold and self.eye_closed:
                    self.blink_count += 1
                    self.eye_closed = False
                    print("blinked")
                    if self.blink_count >= 4:
                        self.blink_led(self.blink_count)
                self.draw_text(self.blink_count)

                for n in range(36, 48):
                    x = landmarks.part(n).x
                    y = landmarks.part(n).y
                    cv2.circle(frame, (x, y), 4, (255, 0, 0), -1)

                # draw the eye aspect ratio and the number of blinks on the frame
                #cv2.putText(frame, "Blinks: {}".format(blink_count), (10, 30),
                            #cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                #cv2.putText(frame, "Eye Aspect Ratio: {:.2f}".format(ear), (300, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # show the frame
            cv2.imshow("Frame", frame)
            input_state = GPIO.input(STOP_BUTTON)
            waitt = cv2.waitKey(1)
            if waitt == ord("q"):
                break
            elif waitt and input_state == False:
                print("PRESSED")
                print("Blink Count: " + str(self.blink_count) + "\nTotal Time: " + str(int(time.time() - self.start_time)))
                self.close_all()
                self.blink_count=0
                self.start_time = time.time()
            
        GPIO.cleanup()
        cv2.destroyAllWindows()
        vs.release()


    def mid_line_distance(self, p1, p2, p3, p4):
        """compute the euclidean distance between the midpoints of two sets of points"""
        p5 = np.array([int((p1[0] + p2[0])/2), int((p1[1] + p2[1])/2)])
        p6 = np.array([int((p3[0] + p4[0])/2), int((p3[1] + p4[1])/2)])

        return norm(p5 - p6)


    def aspect_ratio(self, landmarks, eye_range):
        # Get the eye coordinates
        eye = np.array(
            [np.array([landmarks.part(i).x, landmarks.part(i).y])
            for i in eye_range]
        )
        # compute the euclidean distances
        B = norm(eye[0] - eye[3])
        A = self.mid_line_distance(eye[1], eye[2], eye[5], eye[4])
        # Use the euclidean distance to compute the aspect ratio
        return A / B

    
if __name__ == "__main__":
    blink_detector = BlinkDetector()
    blink_detector.run()


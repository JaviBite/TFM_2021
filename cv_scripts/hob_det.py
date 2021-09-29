# Script para detectar la vitroceramica
# Deteccion de bordes y de figuras ustilizando OpenCV tradicional

import sys, os
import codecs, unicodedata
import cv2, math
import numpy as np

import argparse

def main():

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="Path to the video that will be processed")
    parser.add_argument("output", type=str, nargs="?", default="hub_out.avi", help="Path to save the processed video")

    args = parser.parse_args()

    input_file = args.input
    output_file = args.output

    # Get first video frame
    # Extract video properties
    video = cv2.VideoCapture(input_file)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames_per_second = video.get(cv2.CAP_PROP_FPS)
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize video writer
    video_writer = cv2.VideoWriter(output_file, fourcc=cv2.VideoWriter_fourcc(*"mp4v"), fps=float(frames_per_second), frameSize=(width, height), isColor=True)

    hasFrame = True
    distances = [width**2,width**2,width**2,width**2]
    corners = [(0,0),(0,0),(0,0),(0,0)]
    while True:
        hasFrame, frame = video.read()
        image = frame

        # Show frame
        cv2.imshow("Frame",image)
        #cv2.waitKey(0)

        # Convert to HSV or gray
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = hsv_img[:, :, 0], hsv_img[:, :, 1], hsv_img[:, :, 2]

        #cv2.imshow("HUE", h)
        #cv2.imshow("Saturation", s)
        cv2.imshow("Value", gray_img)

        blurred = cv2.GaussianBlur(v,(5,5),0)
        edges = cv2.Canny(blurred ,50,150,apertureSize = 3)
        cv2.imshow("edges", edges)

        rho = 1  # distance resolution in pixels of the Hough grid
        theta = np.pi / 180  # angular resolution in radians of the Hough grid
        threshold = 100  # minimum number of votes (intersections in Hough grid cell)
        min_line_length = 50  # minimum number of pixels making up a line
        max_line_gap = 20  # maximum gap in pixels between connectable line segments
        line_image = np.copy(image) * 0  # creating a blank to draw lines on

        lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                    min_line_length, max_line_gap)

        lines_edges = image
        if lines is not None:
            for line in lines:
                for x1,y1,x2,y2 in line:
                    angle = math.atan2(y2 - y1 ,x2 - x1)
                    if ((math.pi/2) - abs(angle)) < 0.5:

                        cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),3)

                        # Up left
                        if (distances[0] > math.hypot(0 - x1, 0 - y1)):
                            corners[0] = (x1,y1)
                            distances[0] = math.hypot(0 - x1, 0 - y1)
                        elif (distances[0] > math.hypot(0 - x2, 0 - y2)):
                            corners[0] = (x2,y2)
                            distances[0] = math.hypot(0 - x2, 0 - y2)

                        # Up right
                        if (distances[1] > math.hypot(width - x1, 0 - y1)):
                            corners[1] = (x1,y1)
                            distances[1] = math.hypot(width - x1, 0 - y1)
                        elif (distances[1] > math.hypot(width - x2, 0 - y2)):
                            corners[1] = (x2,y2)
                            distances[1] = math.hypot(width - x2, 0 - y2)

                        # Down left
                        if (distances[3] > math.hypot(0 - x1, height - y1)):
                            corners[3] = (x1,y1)
                            distances[3] = math.hypot(0 - x1, height - y1)
                        elif (distances[3] > math.hypot(0 - x2, height - y2)):
                            corners[3] = (x2,y2)
                            distances[3] = math.hypot(0 - x2, height - y2)

                        # Down right
                        if (distances[2] > math.hypot(width - x1, height - y1)):
                            corners[2] = (x1,y1)
                            distances[2] = math.hypot(width - x1, height - y1)
                        elif (distances[2] > math.hypot(width - x2, height - y2)):
                            corners[2] = (x2,y2)
                            distances[2] = math.hypot(width - x2, height - y2)
            

            for i in range(4):
                p1 = (int(corners[i][0]),int(corners[i][1]))
                p2 = (int(corners[(i+1)%4][0]),int(corners[(i+1)%4][1]))
                cv2.line(line_image,p1,p2,(0,0,255),3)
            

            lines_edges = cv2.addWeighted(image, 0.8, line_image, 1, 0)
            cv2.imshow("Lines",lines_edges)

        video_writer.write(lines_edges)
        
        k = cv2.waitKey(20)
        if k==27:    # Esc key to stop
            break

    # Release resources
    video.release()
    video_writer.release()


if __name__ == '__main__':
    main()
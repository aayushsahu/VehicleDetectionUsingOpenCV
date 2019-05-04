#!/usr/bin/python
import socket
import cv2
import numpy
import random
import sys
from torchvision import transforms
from Libs import Connection, utils
from model import *
from PIL import Image, ImageDraw, ImageFont
import cv2
import matplotlib.pyplot as plt
import numpy as np
import datetime
import traceback
from copy import deepcopy
import pytesseract as tess


host = 'localhost'# sys.argv[1] # e.g. localhost, 192.168.1.123
cam_url = 'rtsp://192.168.1.8:8080/h264_pcm.sdp'       #sys.argv[2] # rtsp://user:pass@url/live.sdp , http://url/video.mjpg ...
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
client_socket.connect((host, 5005))
name = str(cam_url) # gives random name to create window

client_socket.send(str.encode(cam_url))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model checkpoint
checkpoint = 'BEST_checkpoint_ssd300.pth.tar'
checkpoint = torch.load(checkpoint, map_location='cpu')
start_epoch = checkpoint['epoch'] + 1
best_loss = checkpoint['best_loss']
print('\nLoaded checkpoint from epoch %d. Best loss so far is %.3f.\n' % (start_epoch, best_loss))
model = checkpoint['model']
model = model.to(device)
model.eval()

# Transforms
resize = transforms.Resize((300, 300))
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


def preprocess(img):
    cv2.imshow("Input",img)
    print("preprocessing")
    imgBlurred = cv2.GaussianBlur(img, (5,5), 0)
    gray = cv2.cvtColor(imgBlurred, cv2.COLOR_BGR2GRAY)

    sobelx = cv2.Sobel(gray,cv2.CV_8U,1,0,ksize=3)
    cv2.imshow("Sobel",sobelx)
    cv2.waitKey(0)
    ret2,threshold_img = cv2.threshold(sobelx,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    cv2.imshow("Threshold",threshold_img)
    cv2.waitKey(0)
    return threshold_img

def cleanPlate(plate):
    print("CLEANING PLATE. . .")
    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    #kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    #thresh= cv2.dilate(gray, kernel, iterations=1)

    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    contours,hierarchy = cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if contours:
        areas = [cv2.contourArea(c) for c in contours]
        max_index = np.argmax(areas)

        max_cnt = contours[max_index]
        max_cntArea = areas[max_index]
        x,y,w,h = cv2.boundingRect(max_cnt)

        if not ratioCheck(max_cntArea,w,h):
            return plate,None

        cleaned_final = thresh[y:y+h, x:x+w]
        #cv2.imshow("Function Test",cleaned_final)
        return cleaned_final,[x,y,w,h]

    else:
        return plate,None


def extract_contours(threshold_img):
    element = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(17, 3))
    morph_img_threshold = threshold_img.copy()
    cv2.morphologyEx(src=threshold_img, op=cv2.MORPH_CLOSE, kernel=element, dst=morph_img_threshold)
    cv2.imshow("Morphed",morph_img_threshold)
    cv2.waitKey(0)

    contours, hierarchy= cv2.findContours(morph_img_threshold,mode=cv2.RETR_EXTERNAL,method=cv2.CHAIN_APPROX_NONE)
    return contours


def ratioCheck(area, width, height):
    ratio = float(width) / float(height)
    if ratio < 1:
        ratio = 1 / ratio

    aspect = 4.7272
    min = 15*aspect*15  # minimum area
    max = 125*aspect*125  # maximum area

    rmin = 3
    rmax = 6

    if (area < min or area > max) or (ratio < rmin or ratio > rmax):
        return False
    return True

def isMaxWhite(plate):
    avg = np.mean(plate)
    if(avg>=90): # lower the value better is the accuracy while testing 
        return True
    else:
        return False

def validateRotationAndRatio(rect):
    (x, y), (width, height), rect_angle = rect

    if(width>height):
        angle = -rect_angle
    else:
        angle = 90 + rect_angle

    if angle>15:
        return False

    if height == 0 or width == 0:
        return False

    area = height*width
    if not ratioCheck(area,width,height):
        return False
    else:
        return True



def cleanAndRead(img,contours):
    #count=0
    for i,cnt in enumerate(contours):
        min_rect = cv2.minAreaRect(cnt)
        
        print( "in for loop of clean and read")
        if validateRotationAndRatio(min_rect):
            print("if 1")
            x,y,w,h = cv2.boundingRect(cnt)
            plate_img = img[y:y+h,x:x+w]


            if(isMaxWhite(plate_img)):
                #count+=1
                print("if 2")
                clean_plate, rect = cleanPlate(plate_img)

                if rect:
                    print("if 3")
                    x1,y1,w1,h1 = rect
                    x,y,w,h = x+x1,y+y1,w1,h1
                    #cv2.imshow("Cleaned Plate",clean_plate)
                    plate_im = Image.fromarray(clean_plate)
                    text = tess.image_to_string(plate_im, lang='eng')
                    print("Detected Text : ",text)
                    img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
                    img.save('output_images/'+text+'.jpg', "JPEG", quality=100, optimize=True, progressive=True)
                    cv2.imshow("Detected Plate",img)
                    cv2.waitKey(0)

    #print "No. of final cont : " , count

def detect(original_image, min_score, max_overlap, top_k, suppress=None):
    """
    Detect objects in an image with a trained SSD300, and visualize the results.

    :param original_image: image, a PIL Image
    :param min_score: minimum threshold for a detected box to be considered a match for a certain class
    :param max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via Non-Maximum Suppression (NMS)
    :param top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'
    :param suppress: classes that you know for sure cannot be in the image or you do not want in the image, a list
    :return: annotated image, a PIL Image
    """

    # Transform
    image = normalize(to_tensor(resize(original_image)))

    # Move to default device
    image = image.to(device)

    # Forward prop.
    predicted_locs, predicted_scores = model(image.unsqueeze(0))

    # Detect objects in SSD output
    det_boxes, det_labels, det_scores = model.detect_objects(predicted_locs, predicted_scores, min_score=min_score,
                                                             max_overlap=max_overlap, top_k=top_k)

    # Move detections to the CPU
    det_boxes = det_boxes[0].to('cpu')

    # Transform to original image dimensions
    original_dims = torch.FloatTensor(
        [original_image.width, original_image.height, original_image.width, original_image.height]).unsqueeze(0)
    det_boxes = det_boxes * original_dims

    # Decode class integer labels
    det_labels = [rev_label_map[l] for l in det_labels[0].to('cpu').tolist()]

    # If no objects found, the detected labels will be set to ['0.'], i.e. ['background'] in SSD300.detect_objects() in model.py
    if det_labels == ['background']:
        # Just return original image
        return original_image

    # Annotate
    annotated_image = original_image
    draw = ImageDraw.Draw(annotated_image)
    font = ImageFont.truetype("./calibril.ttf", 15)

    # Suppress specific classes, if needed
    for i in range(det_boxes.size(0)):
        if suppress is not None:
            if det_labels[i] in suppress:
                continue

        # Boxes
        box_location = det_boxes[i].tolist()
        draw.rectangle(xy=box_location, outline=label_color_map[det_labels[i]])
        draw.rectangle(xy=[l + 1. for l in box_location], outline=label_color_map[
            det_labels[i]])
        # Text
        text_size = font.getsize(det_labels[i].upper())
        text_location = [box_location[0] + 2., box_location[1] - text_size[1]]
        textbox_location = [box_location[0], box_location[1] - text_size[1], box_location[0] + text_size[0] + 4.,
                            box_location[1]]
        draw.rectangle(xy=textbox_location, fill=label_color_map[det_labels[i]])
        draw.text(xy=text_location, text=det_labels[i].upper(), fill='white',
                  font=font)
    del draw

    return annotated_image


def rcv():
    data = b''
    i=1
    while 1:
        
        try:
            r = client_socket.recv(90456)
            if len(r) == 0:
                exit(0)
            a = r.find(b'END!')
            if a != -1:
                data += r[:a]
                break
            data += r
        except Exception as e:
            print(e)
            continue
    nparr = numpy.fromstring(data, numpy.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if type(frame) is type(None):
        print("none frame received")
        pass
    else:
        try:
            #cv2.imshow(name,frame)
            frame=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame)
            #pil_img.show()
            print(pil_img.size[0],"x",pil_img.size[1])
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            print("Device this deep model is running on: ", device)
            with torch.no_grad():
                output = detect(pil_img, min_score=0.2, max_overlap=0.5, top_k=200)
                pix = np.array(output.getdata()).reshape(output.size[0], output.size[1], 3)
                now=datetime.datetime.now()
                image_name= "["+str(now.year)+"-"+str(now.month)+"-"+str(now.day)+"]["+str(now.hour)+"="+str(now.minute)+"="+str(now.second)+"].jpg"
                output.save('output_images/'+image_name, "JPEG", quality=100, optimize=True, progressive=True)
                #output.show()
                opencv_output = cv2.cvtColor(numpy.array(output), cv2.COLOR_RGB2BGR)
                #output_UMat = cv2.UMat(output)
                #cv2.imshow("DETECTED FRAMES", opencv_output)#_UMat)
                print("FRAME HAS BEEN DISPLAYED")
                print("DETECTING PLATE . . .")

                #img = cv2.imread("testData/Final.JPG")
                img = opencv_output
                threshold_img = preprocess(img)
                contours= extract_contours(threshold_img)
                print("extracted contours")
                #if len(contours)!=0:
                    #print len(contours) #Test
                    # cv2.drawContours(img, contours, -1, (0,255,0), 1)
                    # cv2.imshow("Contours",img)
                    # cv2.waitKey(0)
                cleanAndRead(img,contours)
                print('DONE with a frame')
            if cv2.waitKey(10) == ord('q'):
                client_socket.close()
                sys.exit()
        except:
            print("in except")
            traceback.print_exc()
            client_socket.close()
            exit(0)

while 1:
    rcv()

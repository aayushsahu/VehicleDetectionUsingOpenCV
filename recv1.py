#!/usr/bin/python
import socket
import cv2
import numpy
import random
import sys
import cv2
import numpy as np
import datetime
import traceback
import pytesseract as tess
import torch
import matplotlib.pyplot as plt
import re
import imutils

from copy import deepcopy
from torchvision import transforms
from Libs import Connection, utils
from model import *
from PIL import Image, ImageDraw, ImageFont


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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device this deep model is running on: ", device)

# Transforms
resize = transforms.Resize((300, 300))
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
ip_address_with_portno=cam_url.split('/')[2]
ip_address= ip_address_with_portno.split(':')[0]
filename_array=ip_address.split('.')
filename=filename_array[0]+"_"+filename_array[1]+"_"+filename_array[2]+"_"+filename_array[3]+".jpg"
print(filename)            
firstshot=cv2.imread("first_shot/"+filename) 
#cv2.imshow("firstshot", firstshot)


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
    font = ImageFont.truetype("./calibril.ttf", 40)
    
    for i in range(det_boxes.size(0)):
        if suppress is not None:
            if det_labels[i] in suppress:
                continue
        frame= np.array(original_image)
        # Boxes
        box_location = det_boxes[i].tolist()
        if det_labels[i]=='tvmonitor':    #'bus' or det_labels[i]=='car':
            crop_img = frame[int(box_location[1]):int(box_location[3]), int(box_location[0]):int(box_location[2])]
            #detect_plate()
            plate_no=ocr(crop_img, original_image)
            if plate_no!= None:
                print("VEHICLE CROPPED IMAGE")
                cv2.imshow("Cropped Image", crop_img)
        
    

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
        text_location = [box_location[0] - 10., box_location[1] - text_size[1]]
        textbox_location = [box_location[0], box_location[1] - text_size[1], box_location[0] + text_size[0] + 14.,
                            box_location[1]]
        draw.rectangle(xy=textbox_location, fill=label_color_map[det_labels[i]])
        draw.text(xy=text_location, text=det_labels[i].upper(), fill='white',
                  font=font)
        '''for dims in box_location:
            print("DIM: ",int(dims))
        print('\n')'''    
                
    del draw
    return annotated_image

def ocr(plate_image, frame):
	try:
		im_RGB=plate_image[..., ::-1]
		print("plate rgb: ", type(im_RGB))
		#im_RGB=cv2.image()
		firstshot_array=np.array(firstshot)
		print("first shot dims:",firstshot_array.ndim)
		frame_array=np.array(frame)
		print("frame dims:", frame_array.ndim)
		#cv2.imshow("FRAME_ARRAY_BEFORE", frame_array)
		for i in range(0, len(frame_array)-1):
			for j in range(0, len(frame_array[i])-1):
				frame_array[i, j, 0]=frame_array[i, j, 0]^firstshot_array[i, j, 0]
				frame_array[i, j, 1]=frame_array[i, j, 1]^firstshot_array[i, j, 1]
				frame_array[i, j, 2]=frame_array[i, j, 2]^firstshot_array[i, j, 2]
		cv2.imwrite("frame_array.jpg", frame_array)
		#cv2.imshow("FRAME_ARRAY_JPG", frame_array)

		gray = cv2.cvtColor(im_RGB, cv2.COLOR_RGB2GRAY) #to grayscale
		img_blur = cv2.bilateralFilter(gray, 11, 17, 17)
		#img_blur = cv2.GaussianBlur(grey,(5,5),0)
		#img_blur = cv2.medianBlur(grey, 5)    # no. must be always odd
		#img_thresh_Gaussian = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
		#plt.imshow(img_thresh_Gaussian)
		edged = cv2.Canny(img_blur, 30, 200) #Perform Edge detection
		#cv2.imshow("After CANNY", edged)
		nts = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		cnts = imutils.grab_contours(nts)
		cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]
		screenCnt = None
		# loop over our contours
		for c in cnts:
			# approximate the contour
			peri = cv2.arcLength(c, True)
			approx = cv2.approxPolyDP(c, 0.018 * peri, True)
			# if our approximated contour has four points, then
			# we can assume that we have found our screen
			if len(approx) == 4:
				screenCnt = approx
				break
		# Masking the part other than the number plate
		mask = np.zeros(gray.shape,np.uint8)
		new_image = cv2.drawContours(mask,[screenCnt], 0,255,-1,)
		new_image = cv2.bitwise_and(plate_image, plate_image, mask=mask)
		
		# Now crop
		(x, y) = np.where(mask == 255)
		#cv2.imshow("Vehicle Image", new_image)
		cv2.waitKey(0)
		try:
			(topx, topy) = (np.min(x), np.min(y))
			(bottomx, bottomy) = (np.max(x), np.max(y))
			cropped = gray[topx:bottomx+1, topy:bottomy+1]

		 	#Read the number plate
			text = tess.image_to_string(cropped, config='--psm 11')
			output_window =np.hstack((frame, cropped))
			cv2.imshow('output_window', output_window)
			#cv2.imshow("Cropped", cropped)
			plate_no=tess.image_to_string(img_thresh_Gaussian, lang='eng')
			#print(plate_no)
			cv2.imshow("im_RGB", im_RGB)
			regex=r'[A-Za-z][A-Za-z]\s*[0-9][0-9]*\s*[A-Za-z]+\s*[0-9][0-9][0-9][0-9]'
			flags=0
			if re.compile(regex, flags).match(plate_no):
				print("Detected Number is:",text)
				im_RGB = cv2.cvtColor(plate_image, cv2.COLOR_BGR2RGB)
				im_pil = Image.fromarray(im_RGB)
				im_pil.save('plate_images/'+plate_no, "JPEG", quality=100, optimize=True, progressive=True)
				return plate_no
			else:
				return None
			return plate_no
		except:
			return None
	except:
		return None

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
            pass
    nparr = numpy.fromstring(data, numpy.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if type(frame) is type(None):
        print("none frame received")
        pass
    else:
        try:
            #cv2.imshow(name,frame)
            frame=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            #sub_frame=frame_array #cv2.imwrite("temp.jpg", frame_array)          			
            #print("sun_frame=", sub_frame.dtype)
            pil_img = Image.fromarray(frame)
            #pil_img.show()
            print(pil_img.size[0],"x",pil_img.size[1])
            with torch.no_grad():
                output = detect(pil_img, min_score=0.2, max_overlap=0.5, top_k=200)
                pix = np.array(output.getdata()).reshape(output.size[0], output.size[1], 3)
                now=datetime.datetime.now()
                image_name= "["+str(now.year)+"-"+str(now.month)+"-"+str(now.day)+"]["+str(now.hour)+"="+str(now.minute)+"="+str(now.second)+"].jpg"
                output.save('output_images/'+image_name, "JPEG", quality=100, optimize=True, progressive=True)
                #output.show()
                opencv_output = cv2.cvtColor(numpy.array(output), cv2.COLOR_RGB2BGR)
                #output_UMat = cv2.UMat(output)
                #cv2.imshow("Detected Frames", opencv_output)
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

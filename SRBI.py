import cv2
import socket
import sys
import numpy as np
import traceback
from PIL import Image, ImageDraw, ImageFont


host = 'localhost'# sys.argv[1] # e.g. localhost, 192.168.1.123
cam_url = 'rtsp://192.168.1.8:8080/h264_pcm.sdp'       #sys.argv[2] # rtsp://user:pass@url/live.sdp , http://url/video.mjpg ...
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
client_socket.connect((host, 5005))
name = str(cam_url) # gives random name to create window
client_socket.send(str.encode(cam_url))

def rcv():
    data = b''
    i=1
    iteration=1
    fgbg = cv2.createBackgroundSubtractorMOG2()
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
    nparr = np.fromstring(data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if type(frame) is type(None):
        print("none frame received")
        pass
    else:
        try:
            #for creating filename
            
            frame=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_temp=frame
            if iteration>1:
                frame_prev=cv2.imread("first_shot/"+filename)
                for i in range(len(frame)):
                    for j in range(len(frame[0])):
                        if frame_temp[i, j] != frame_prev[i, j]:
                            frame_temp[i, j]=0

            cv2.imshow("frame_temp", frame_temp)

            ip_address_with_portno=cam_url.split('/')[2]
            ip_address= ip_address_with_portno.split(':')[0]
            filename_array=ip_address.split('.')
            filename=filename_array[0]+"_"+filename_array[1]+"_"+filename_array[2]+"_"+filename_array[3]+".jpg"
            print(filename)

            frame_BGR = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
            pil_img = Image.fromarray(frame_BGR)
            pil_img.save('first_shot/'+filename, "JPEG", quality=100, optimize=True, progressive=True)








            '''pil_img = Image.fromarray(frame)
            pil_img.show()
            print(pil_img.size[0],"x",pil_img.size[1])
            #pix = np.array(pil_img.getdata()).reshape(frame.size[0], frame.size[1], 3)
            
            frame_BGR = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
            cv2.imshow("First Shot", frame_BGR)'''
            if cv2.waitKey(10) == ord('q'):
                client_socket.close()
                sys.exit()
        except:
        	print("in except")
        	traceback.print_exc()
        finally:
        	client_socket.close()
        	exit(0)
while True:
    rcv()
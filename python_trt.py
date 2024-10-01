from ctypes import *
import cv2
import numpy as np
import numpy.ctypeslib as npct
import time
import math
import threading
from PIL import Image

###新增
import keyboard
import os
import pygame
from gtts import gTTS
from mutagen.mp3 import MP3
from PIL import ImageDraw, ImageFont
###

###新增
twocamera=0
t0=0
t01=0
pasue=0
temperary=0
video_path = "0"
###

###新增
def gstreamer_pipeline1(
    sensor_id=1,
    capture_width=1280,
    capture_height=720,
    display_width=640,
    display_height=360,
    framerate=30,
    flip_method=0,
):
    return (
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )
def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1280,
    capture_height=720,
    display_width=640,
    display_height=360,
    framerate=30,
    flip_method=0,
):
    return (
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )
def get_mp3_duration(file_path):
    try:
        audio = MP3(file_path)
        duration_in_seconds = audio.info.length
        return duration_in_seconds
    except Exception as e:
        print(f"Error: {e}")
        return None
    
def start_process():
    global running
    running = not running
def quit_program():
    global running
    running = False
    keyboard.unhook_all()
    exit()

running = False

keyboard.add_hotkey('s', start_process)  # Bind Ctrl+Shift+S to start/pause the process
keyboard.add_hotkey('q', quit_program)   # Bind Ctrl+Shift+Q to quit the program

###

# -----------------------------------双目相机的基本参数---------------------------------------------------------
#   left_camera_matrix          左相机的内参矩阵
#   right_camera_matrix         右相机的内参矩阵
#
#   left_distortion             左相机的畸变系数    格式(K1,K2,P1,P2,0)
#   right_distortion            右相机的畸变系数
# -------------------------------------------------------------------------------------------------------------
# 左镜头的内参，如焦距
left_camera_matrix = np.array([[516.5066236,-1.444673028,320.2950423],[0,516.5816117,270.7881873],[0.,0.,1.]])
right_camera_matrix = np.array([[511.8428182,1.295112628,317.310253],[0,513.0748795,269.5885026],[0.,0.,1.]])

# 畸变系数,K1、K2、K3为径向畸变,P1、P2为切向畸变
left_distortion = np.array([[-0.046645194,0.077595167, 0.012476819,-0.000711358,0]])
right_distortion = np.array([[-0.061588946,0.122384376,0.011081232,-0.000750439,0]])

# 旋转矩阵
R = np.array([[0.999911333,-0.004351508,0.012585312],
              [0.004184066,0.999902792,0.013300386],
              [-0.012641965,-0.013246549,0.999832341]])
# 平移矩阵
T = np.array([-120.3559901,-0.188953775,-0.662073075])

###範例 有鏡頭/有影片
#capture = cv2.VideoCapture(gstreamer_pipeline(flip_method=0),cv2.CAP_GSTREAMER)
#capture = cv2.VideoCapture("car.avi")

###新增鏡頭
if(twocamera==1):
    capture = cv2.VideoCapture(gstreamer_pipeline(flip_method=0),cv2.CAP_GSTREAMER)
    capture2 = cv2.VideoCapture(gstreamer_pipeline1(flip_method=0),cv2.CAP_GSTREAMER)
    ref2, frame2 = capture2.read()
else:
    capture = cv2.VideoCapture("car.avi")

ref, frame = capture.read()

if not ref:
    raise ValueError("未能正确读取摄像头（视频），请注意是否正确安装摄像头（是否正确填写视频路径）。")
if(twocamera==1 and not ref2): 
    raise ValueError("未能正确读取摄像头（视频），请注意是否正确安装摄像头（是否正确填写视频路径）。")
###

#長
width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
if(twocamera == 0):
    width = width//2

#寬
height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(width)
print(height)

size = (640, 480)

R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(left_camera_matrix, left_distortion,
                                                                  right_camera_matrix, right_distortion, size, R,
                                                                  T)
# 校正查找映射表,将原始图像和校正后的图像上的点一一对应起来
left_map1, left_map2 = cv2.initUndistortRectifyMap(left_camera_matrix, left_distortion, R1, P1, size, cv2.CV_16SC2)
right_map1, right_map2 = cv2.initUndistortRectifyMap(right_camera_matrix, right_distortion, R2, P2, size, cv2.CV_16SC2)
print(Q)

# ---------------------------------------------------------------------------------------------------------
#   classes        coco数据集的种类,网络返回‘0’时，对应着person，依次类推
# ---------------------------------------------------------------------------------------------------------
classes = ('person','bicycle','car','motorbike','aeroplane','bus','train','truck','boat','traffic light',
'fire hydrant','stop sign','parking meter','bench','bird','cat','dog','horse','sheep','cow','elephant',
'bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase','frisbee','skis','snowboard',
'sports ball','kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket','bottle',
'wine glass','cup','fork','knife','spoon','bowl','banana','apple','sandwich','orange','broccoli','carrot',
'hot dog','pizza','donut','cake','chair','sofa','pottedplant','bed','diningtable','toilet','tvmonitor',
'laptop','mouse','remote','keyboard','cell phone','microwave','oven','toaster','sink','refrigerator',
'book','clock','vase','scissors','teddy bear','hair drier','toothbrush')

cla = ('人', '自行車', '汽車', '摩托車', '飛機', '巴士', '火車', '卡車', '船', '紅綠燈', 
'消防栓', '停車標誌', '停車收費表', '長椅', '鳥', '貓', '狗', '馬', '羊', '牛', '大象', 
'熊', '斑馬', '長頸鹿', '背包', '雨傘', '手提包', '領帶', '行李箱', '飛盤', '滑雪板', '滑雪板', 
'運動球', '風箏', '棒球棒', '棒球手套', '滑板', '衝浪板', '網球拍', '瓶子', 
'酒杯', '杯子', '叉子', '刀', '湯匙', '碗', '香蕉', '蘋果', '三明治', '橙子', '西蘭花', '胡蘿蔔', 
'熱狗', '披薩', '甜甜圈', '蛋糕', '椅子', '沙發', '盆栽植物', '床', '餐桌', '廁所', '電視機', 
'筆記本電腦', '滑鼠', '遙控器', '鍵盤', '手機', '微波爐', '烤箱', '烤麵包機', '水槽', '冰箱', 
'書', '時鐘', '花瓶', '剪刀', '泰迪熊', '吹風機', '牙刷')


# ---------------------------------------------------------------------------------------------------------
#   Detector()       配置tensorrt加速
# ---------------------------------------------------------------------------------------------------------
class Detector():
    def __init__(self,model_path,dll_path):
        self.yolov5 = CDLL(dll_path,winmode=0)
        self.yolov5.Detect.argtypes = [c_void_p,c_int,c_int,POINTER(c_ubyte),npct.ndpointer(dtype = np.float32, ndim = 2, shape = (50, 6), flags="C_CONTIGUOUS")]
        self.yolov5.Init.restype = c_void_p
        self.yolov5.Init.argtypes = [c_void_p]
        self.yolov5.cuda_free.argtypes = [c_void_p]
        self.c_point = self.yolov5.Init(model_path)

    def predict(self,img):
        rows, cols = img.shape[0], img.shape[1]
        res_arr = np.zeros((50,6),dtype=np.float32)
        self.yolov5.Detect(self.c_point,c_int(rows), c_int(cols), img.ctypes.data_as(POINTER(c_ubyte)),res_arr)
        self.bbox_array = res_arr[~(res_arr==0).all(1)]
        return self.bbox_array

    def free(self):
        self.yolov5.cuda_free(self.c_point)

# ------------------------------------visualize可视化程序-------------------------------------------------------
#   img                     输入的图片
#   bbox_array              多组yolo网络预测的结果
#   middle_x、middle_y       检测目标的中心点坐标，用于测出距离distance
# -------------------------------------------------------------------------------------------------------------
def visualize(img,bbox_array):
    global dism
    global clasm
    global Dir
    dism = float('inf')
    clasm = 0
    
    for temp in bbox_array:
        bbox = [temp[0],temp[1],temp[2],temp[3]]  #xywh
        clas = int(temp[4])
        score = temp[5]
        #精準度(置信度被放在dll檔，要改要重新編譯，改yolo.cpp CONF_THRESH 0.5)
        #if(score>=0.5):
            #print(score)

        #顯示種類 -> str(classes[int(clas)])
        #print(str(classes[int(clas)]))

        middle_x = int(np.floor(temp[0]+temp[2]*0.5))
        middle_y = int(np.floor(temp[1]+temp[3]*0.5))


        ##新增方向 Dir (car.avi適用) --> 鏡頭適用 請用(twocamera=0)
        ### 左前 正前 右前
        if middle_x < width//3:   #  // -> 整除
            Dir = "左前"
            Dire = "left"
        elif (width//3 < middle_x) and (middle_x < width//3*2):
            Dir = "正前"
            Dire = "middle"
        elif middle_x > width//3*2:
            Dir = "右前"
            Dire = "right"
        
        ###

        
        distance = math.sqrt(threeD[middle_y][middle_x][0] ** 2 +
                             threeD[middle_y][middle_x][1] ** 2 + threeD[middle_y][middle_x][2] ** 2)
        distance = distance / 1000.0  # mm -> m
        if dism > distance:
            dism = round(distance,1)#四捨五入取小數點第1位
            clasm = int(temp[4])
        #顯示距離 取小數點第二位 -> str(round(distance,2))
        ##單位公尺
        #print(str(round(distance,2)))
        
        cv2.rectangle(img,(int(temp[0]),int(temp[1])),(int(temp[0]+temp[2]),int(temp[1]+temp[3])), (0, 0, 225), 2)
        img = cv2.putText(img, classes[int(clas)] +" "+str(round(score,2))+" dis="+str(round(distance,2))+" "+Dire,
                          (int(temp[0]),int(temp[1])-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 225), 2)
    dism = str(dism)
    #print()
    #print(dism + classes[int(clasm)])


    ##新增分割線(car.avi適用) --> 鏡頭適用 請用(twocamera=0)
    ### 左 中 右分割線
    cv2.line(img, (width//3, 0), (width//3, height), (0, 255, 0), 1)

    cv2.line(img, (width//3*2, 0), (width//3*2, height), (0, 255, 0), 1)

    

    return img

#---------------------------------------------------#
#   对输入图像进行不失真resize
#---------------------------------------------------#
def resize_image(image, size):
    iw, ih  = image.size
    w, h    = size
    scale   = min(w/iw, h/ih)
    nw      = int(iw*scale)
    nh      = int(ih*scale)

    image   = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    return new_image

#----------------------加载推理所需要的engine、dll文件---------------------------------#
#   yolov5s_480         图片推理的输入格式为640x480
#   yolov5              图片推理的输入格式为640x640，若使用它，需要打开182行的resize函数
#------------------------------------------------------------------------------------#
det = Detector(model_path=b"./yolov5s.engine",dll_path=r"C:\Users\user\Desktop\a\Jeston nano_tensorrt\部署\yolov5.dll")  # b'' is needed
# 加载视频文件
#capture = cv2.VideoCapture("car.avi")
WIN_NAME = 'Deep disp'
cv2.namedWindow(WIN_NAME, cv2.WINDOW_AUTOSIZE)

# 3 读取视频
fps = 0.0
#ret, frame = capture.read()

while ref:
    # 是否读取到了帧，读取到了则为True
    ref, frame = capture.read()
    if(twocamera==1): 
        ref2, frame2 = capture2.read()

    if not ref:
        break
    if(twocamera==1 and not ref2): 
        break
    
    # 开始计时，用于计算帧率
    t1 = time.time()
    # 格式转变，BGRtoRGB
    img_color = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if(twocamera==1):
        img_color2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)

    # 割开双目图像
    if(twocamera == 1):
        frame1 = frame[0:480, 0:640]
        frame2 = frame2[0:480, 0:640]  
    else:        
        frame1 = frame[0:480, 0:640]
        frame2 = frame[0:480, 640:1280]
        ##frame2 = frame[0:480, 320:640]
    

    imgL = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)  # 将BGR格式转换成灰度图片
    imgR = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # cv2.remap 重映射，就是把一幅图像中某位置的像素放置到另一个图片指定位置的过程。
    # 依据MATLAB测量数据重建无畸变图片
    img1_rectified = cv2.remap(imgL, left_map1, left_map2, cv2.INTER_LINEAR)
    img2_rectified = cv2.remap(imgR, right_map1, right_map2, cv2.INTER_LINEAR)

    imageL = cv2.cvtColor(img1_rectified, cv2.COLOR_GRAY2BGR)
    imageR = cv2.cvtColor(img2_rectified, cv2.COLOR_GRAY2BGR)

    # ------------------------------------SGBM算法----------------------------------------------------------
    #   blockSize                   深度图成块，blocksize越低，其深度图就越零碎，0<blockSize<10
    #   img_channels                BGR图像的颜色通道，img_channels=3，不可更改
    #   numDisparities              SGBM感知的范围，越大生成的精度越好，速度越慢，需要被16整除，如numDisparities
    #                               取16、32、48、64等
    #   mode                        sgbm算法选择模式，以速度由快到慢为：STEREO_SGBM_MODE_SGBM_3WAY、
    #                               STEREO_SGBM_MODE_HH4、STEREO_SGBM_MODE_SGBM、STEREO_SGBM_MODE_HH。精度反之
    # -----------------------------------------------------------------------------------------------------
    blockSize = 8
    img_channels = 3
    stereo = cv2.StereoSGBM_create(minDisparity=1,
                                   numDisparities=32,
                                   blockSize=blockSize,
                                   P1=8 * img_channels * blockSize * blockSize,
                                   P2=32 * img_channels * blockSize * blockSize,
                                   disp12MaxDiff=-1,
                                   preFilterCap=1,
                                   uniquenessRatio=10,
                                   speckleWindowSize=100,
                                   speckleRange=100,
                                   mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    # 计算视差
    disparity = stereo.compute(img1_rectified, img2_rectified)
    # 归一化函数算法
    disp = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # 计算三维坐标数据值
    threeD = cv2.reprojectImageTo3D(disparity, Q, handleMissingValues=True)
    threeD = threeD * 16

    # 格式转变，BGRtoRGB
    frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
    # 转变成Image格式
    frame1 = Image.fromarray(np.uint8(frame1))
    frame1_shape = np.array(np.shape(frame1)[0:2])
    # 调整图片大小、颜色通道，使其适应YOLO推理的格式
    # frame1 = resize_image(frame1,(640,480))
    frame1 = cv2.cvtColor(np.array(frame1), cv2.COLOR_RGB2BGR)

    # 推理图片
    result = det.predict(frame1)
    ##result = det.predict(frame)
    #print(result)
    
    # 画框，标出识别的类别、距离、置信度等
    frame1 = visualize(frame1, result)
    ##frame1 = visualize(frame, result)

    ### 新增語音, 說出目前最近的物體, 不會搶快講下一個(但需要pip install pygame)
    ### pb:1.距離檢測不到會說inf 2.沒物件在畫面按播放不會crash(應該)
    ###    
    pygame.mixer.init()
    if running == True and pygame.mixer.music.get_busy() == False :
        pygame.mixer.quit()
        #print(t0)
        #print(t01)
        #print(time.time())
        #print(int(time.time()-t0)//7)

        t0 = 0
        t01 = 0
        
        label = str(cla[int(clasm)])
        if label!=0:
            mytext = label + "距離" + Dir + "方" + dism + "公尺"

            if dism != 'inf':
                # Language we want to use 
                language = 'zh-tw'
                myobj = gTTS(text=mytext, lang=language, slow=True)
                myobj.save("output.mp3")
                #time.sleep(0.5)

                file_path = "output.mp3"  # Path to your MP3 file
                duration = get_mp3_duration(file_path)
                if duration is not None:
                    pygame.mixer.init()
                    pygame.mixer.music.load("output.mp3")
                    
                    # 播放音頻
                    pygame.mixer.music.play()

                    # 使用判斷式檢查是否正在播放
                    if pygame.mixer.music.get_busy():
                        print("音檔正在播放中")
                    else:
                        print("音檔已停止播放")
                    # Play the converted file 
                    #os.system("mpg123 output.mp3")
                    #os.system("output.mp3")

    ###
    # 计算帧率
    fps = (fps + (1. / (time.time() - t1))) / 2
    print("fps= %.2f" % (fps))
    frame1 = cv2.putText(frame1, "fps= %.2f" % (fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow("frame1", frame1)

    cv2.imshow(WIN_NAME, disp)  # 显示深度图的双目画面
    # 若键盘按下q则退出播放
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

# 4 释放资源
det.free()
capture.release()

# 5 关闭所有窗口
cv2.destroyAllWindows()

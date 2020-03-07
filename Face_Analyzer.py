#Libraries
################################
import numpy as np
import cv2 as cv
import glob
import threading
import pickle
from sklearn.svm import SVC
from imutils import face_utils
import imutils
import dlib
from collections import OrderedDict 
################################
 
#Defining Macros
################################
#Common Tweaks

RESIZE_HEIGHT = 96
RESIZE_WIDTH = 96

HEIGHT_THRESH = 300
WIDTH_THRESH = 300

POS_IMAGES = "pos/*.jpg"
NEG_IMAGES = "neg/*.png"
NEG_IMAGES_2 = "negative/*.jpg"

test_image = 'abc.jpg'

filename = 'trained_svm.sav'

################################
#Relative Positions To Root Part

FACIAL_LANDMARKS_IDXS = OrderedDict([
	("mouth", (48, 68)),
	("right_eyebrow", (17, 22)),
	("left_eyebrow", (22, 27)),
	("right_eye", (36, 42)),
	("left_eye", (42, 48)),
	("nose", (27, 35)),
	("jaw", (0, 17))
])
################################

################################
#HOG Configuration
BLOCK_SIZE = (16,16)
BLOCK_STRIDE = (8,8)
CELL_SIZE = (8,8)
NBINS = 9

def computeHOG(image):
    rows, cols = image.shape
    WINDOW_SIZE = (cols, rows)
    hog = cv.HOGDescriptor(WINDOW_SIZE,BLOCK_SIZE,BLOCK_STRIDE,CELL_SIZE,NBINS)
    output = hog.compute(image)
    return output.flatten()

################################
    
def getImages(IMAGES_DIR):
    
    cv_img = []
    for img in glob.glob(IMAGES_DIR):
        n = cv.imread(img, 0)
        cv_img.append(n)
        
    return cv_img

def getNegImages(IMAGES_DIR):
    
    cv_img = []
    for img in glob.glob(IMAGES_DIR):
        cv_img.append(cv.resize(cv.imread(img, 0), (RESIZE_HEIGHT, RESIZE_WIDTH)))
        
    for img in glob.glob(NEG_IMAGES_2):
        cv_img.append(cv.resize(cv.imread(img, 0), (RESIZE_HEIGHT, RESIZE_WIDTH)))
    
    return cv_img

train_root = []
def trainSVM():
    
    imgs = getImages(POS_IMAGES)
    neg = getNegImages(NEG_IMAGES)
    neg_len = len(neg)
    pos_len = len(imgs)
    
    print("Training Root SVM")
    print("Amount of negative images: ", neg_len)
    print("Amount of positive images: ", pos_len)
    
    labels_root = np.hstack((np.zeros(neg_len), np.ones(pos_len)))
     
    for i in neg:
        train_root.append(computeHOG(i))

    for i in imgs:
        train_root.append(computeHOG(i))

    svm_root.fit(train_root, labels_root)

    print("Done Collecting Training Data")    


def selectiveSearch(im):
    
    # speed-up using multithreads
    cv.setUseOptimized(True);
    cv.setNumThreads(4);
          
    ss = cv.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(im)
     
    # Switch to fast but low recall Selective Search method
    #ss.switchToSelectiveSearchFast()
    # Switch to high recall but slow Selective Search method
    ss.switchToSelectiveSearchQuality()

    boxes = ss.process()
    print('Selective Search: Number of Region Proposed: {}'.format(len(boxes)))
    
    return boxes


def calculateIoU(xmin, ymin, xmax, ymax, x_min, y_min, x_max, y_max):
    
    x1 = max(xmin, x_min)
    y1 = max(ymin, y_min)
    x2 = min(xmax, x_max)
    y2 = min(ymax, y_max)
    
    area_box1 = (xmax - xmin+1) * (ymax - ymin+1)
    area_box2 = (x_max - x_min+1) * (y_max - y_min+1)    
    
    area_intersection = max(0, x2-x1+1) * max(0, y2-y1+1)
    iou = area_intersection/float(area_box1 + area_box2 - area_intersection)
    return iou


def nonMaximumSuppression(detections, overlapThresh):
    
	if len(detections) == 0:
		return []
 

	boxes = np.asarray(detections, dtype=np.float)

	pick = []
 
	x1 = boxes[:,0]
	y1 = boxes[:,1]
	x2 = boxes[:,2]
	y2 = boxes[:,3]
 
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = np.argsort(y2)
 
	while len(idxs) > 0:

		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)
 
		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])
 
		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)
 
		# compute the ratio of overlap
		overlap = (w * h) / area[idxs[:last]]
 
		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > overlapThresh)[0])))
 
	return boxes[pick].astype("int")


detections = []
#confidences = []
lock = threading.Lock()
test_hog = []

def detect(bnd_box, image):
    
    for box in bnd_box:
        x, y, x_max, y_max = box
        x_max, y_max = x+x_max, y+y_max
        box2 = x, y, x_max, y_max
        resized_image = cv.resize(image[y:y_max, x:x_max], (RESIZE_HEIGHT,RESIZE_WIDTH))
        local_HOG = computeHOG(resized_image).reshape(1, -1)
        test_hog.append(local_HOG)
        
        detection = svm_root.predict(local_HOG)
        
        #Synchronization
        if(detection == 1):
            lock.acquire()
            detections.append(box2)
            lock.release()
        

def matchHOG(image):

    test = cv.imread(test_image, 1)
    bnd_box = selectiveSearch(test)
    length = len(bnd_box)
    chunk = int(length/4) #Dividing Regions
    
    #Multithreading:
    t1 = threading.Thread(target=detect, args=(bnd_box[0:chunk], image))
    t2 = threading.Thread(target=detect, args=(bnd_box[chunk:chunk*2], image))
    t3 = threading.Thread(target=detect, args=(bnd_box[chunk*2:chunk*3], image))  
    t4 = threading.Thread(target=detect, args=(bnd_box[chunk*3:length], image))  
    t1.start()
    t2.start()
    t3.start()
    t4.start()
    t1.join() 
    t2.join()
    t3.join() 
    t4.join() 


def calculatePenalty(tuple_orig, tuple_detect):
    
    deviation_x = abs(tuple_orig[0]-tuple_detect[0])
    deviation_y = abs(tuple_orig[1]-tuple_detect[1])
    
    a = 0.15
    b = 0.35
    c = 0.15
    d = 0.35
    
    penalty = (a*deviation_x)**2+(b*deviation_x)+(c*deviation_y)**2+(d*deviation_y)
    return penalty

def saveSVM():
    pickle.dump(svm_root, open(filename, 'wb'))


def calculateGoldenRatioAndSymmetry(shape, img):
    
    image = img.copy()
    
    #Nose
    (n1, n2) = shape[33]
    
    #Left-Eye
    (p1, p2) = shape[37]
    (p3, p4) = shape[40]
    pupil_y = int((p2+p4)/2)
    
    (p1, p2) = shape[36]
    (p3, p4) = shape[39]
    pupil_x = int((p1+p3)/2)
    
    #Right-Eye
    (p1, p2) = shape[43]
    (p3, p4) = shape[46]
    pupil_y1 = int((p2+p4)/2)
    
    (p1, p2) = shape[42]
    (p3, p4) = shape[45]
    pupil_x1 = int((p1+p3)/2)
    
    euclid_dist_eye = ((pupil_y1-pupil_y)**2 + (pupil_x1-pupil_x)**2)**0.5
    
    #Mouth-Left
    (m1, m2) = shape[48]
    (m3, m4) = shape[54]
    
    
    #Face Width
    (f1, f2) = shape[0]
    (f3, f4) = shape[16]
    
    euclid_dist_jaw = ((f4-f2)**2 + (f3-f1)**2)**0.5
    
    cv.line(image,(pupil_x,pupil_y),(pupil_x1,pupil_y1),(255,0,0), 1) #Eyes
    cv.line(image,(n1,n2),(pupil_x,pupil_y),(255,0,0), 1) #Nose-Left Eye
    cv.line(image,(n1,n2),(pupil_x1,pupil_y1),(255,0,0), 1) #Nose-Right Eye
    cv.line(image,(n1,n2),(f1,f2),(255,0,0), 1) #Nose-Jaw
    cv.line(image,(n1,n2),(f3,f4),(255,0,0), 1) #Nose-Jaw    
    
    copy = image.copy()    

    goldRatio = round(euclid_dist_jaw/euclid_dist_eye, 2)
    print("Golden Ratio: "+str(goldRatio))
    
    cv.putText(image, "Golden Ratio:"+str(goldRatio), (5, 20),
    cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    cv.imwrite("face_"+str(goldRatio)+'.jpg', image)
    
    
    ###########Symmetry#############
    
    #Nose-Length
    (n3, n4) = shape[31]
    (n5, n6) = shape[35]
    
    #Nose-Eye
    euclid_dist_1 = ((n1-pupil_x)**2 + (n2-pupil_y)**2)**0.5 
    euclid_dist_2 = ((n1-pupil_x1)**2 + (n2-pupil_y1)**2)**0.5
    
    penalty_1 = penaltyCalculator(euclid_dist_1, euclid_dist_2)
    
    #Nose-Nose Width
    euclid_dist_3 = ((n1-f1)**2 + (n2-f2)**2)**0.5
    euclid_dist_4 = ((n1-f3)**2 + (n2-f4)**2)**0.5
    
    penalty_2 = penaltyCalculator(euclid_dist_3, euclid_dist_4)
    
    #Nose-Mouth
    euclid_dist_5 = ((n1-m1)**2 + (n2-m2)**2)**0.5
    euclid_dist_6 = ((n1-m3)**2 + (n2-m4)**2)**0.5
    
    penalty_3 = penaltyCalculator(euclid_dist_5, euclid_dist_6)
    
    #Weights of each penalty
    #w1, w2, w3 = 0.5, 0.15, 0.35
    #symmetry = 1 - (penalty_1*w1)+(penalty_2*w2)+(penalty_3*w3)
    
    symmetry = 1-(penalty_1+penalty_2+penalty_3)
    
    #Percentage
    symmetry = round(symmetry * 100, 2)
    
    print("Face Symmetry: ", symmetry, '%')
    cv.putText(copy, "Symmetry: "+str(symmetry), (5, 20),
    cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    cv.line(image,(n1,n2),(f1,f2),(255,0,0), 1) #Nose-Jaw
    cv.line(image,(n1,n2),(f3,f4),(255,0,0), 1) #Nose-Jaw 
    cv.line(copy,(n1,n2),(m1,m2),(255,0,0), 1) #Nose-Mouth
    cv.line(copy,(n1,n2),(m3,m4),(255,0,0), 1) #Nose-Mouth
    cv.imwrite("faceSym_"+str(symmetry)+'.jpg', copy)
    
def penaltyCalculator(e1, e2):
    
    if(e1 < e2):
        return (1-(e1/e2))
    else:
        return (1-(e2/e1))

    
def landmarkDetection(rects, image):
    
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
    rects = detector(gray, 1)
    
    for (i, rect) in enumerate(rects):
    
    	shape = predictor(gray, rect)
    	shape = face_utils.shape_to_np(shape)
     
    	calculateGoldenRatioAndSymmetry(shape, image)
    	break
                


#Main Function Below

test = cv.imread(test_image, 0)
rows, cols = test.shape


######### For trained SVM, load the model from disk #########
svm_root = pickle.load(open(filename, 'rb'))   

#svm_root = SVC(kernel='linear')
#trainSVM()
#saveSVM()

matchHOG(test)

picks = nonMaximumSuppression(detections, 0.5)

output = cv.imread(test_image, 1)

for i in picks:    #Send Detection
    landmarkDetection(i, output[i[1]:i[3], i[0]:i[2]])

for i in picks:
    x_y = (i[0], i[1])
    x1_y1 = (i[2], i[3])
    cv.rectangle(output, x_y, x1_y1, color=(0, 255, 0), thickness=2)

cv.imwrite('output.jpg', output)







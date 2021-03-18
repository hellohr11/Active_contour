#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import cv2
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


def captureVid():
    # define a video capture object 
    vid = cv2.VideoCapture(0) 

    while(True): 
        ret, frame = vid.read()  
        cv2.imshow('frame', frame) 
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            vid.release()
            cv2.destroyAllWindows() 
            return frame
        
def showImg(img):
    #show image
    #b=cv2.resize(img,(800,600))
    cv2.imshow('image',img)  
    
def grayImg(img):
    #convert image to graycolor
    imgp = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return imgp

def noise(img):
    #get a noisy image
    img1=img.copy()
    img1=cv2.randn(img1,(100),(200))
    images=cv2.add(img, img1) 
    return images

def smooth(img):
    global imgp
    n=5
    kernel = np.ones((n,n), np.float32)/(n*n)
    imgp = cv2.filter2D(img, -1, kernel)
    
    
def gradient(img):
    #get derivative of gradient in x and y direction, results are normalized to (0,255)
    sx=cv2.Sobel(img,cv2.CV_64F,1,0,ksize=31)
    sy=cv2.Sobel(img,cv2.CV_64F,0,1,ksize=31)
    
    mag, ang = cv2.cartToPolar(sx, sy)
    """

    sx=sx/mag.max()
    sy=sy/mag.max()
    """
    mag=(mag-mag.min())/(mag.max()-mag.min())
    gx,gy=np.gradient(mag)
    
    

    return gx,gy


# In[3]:


def createA(a, b, N):
    #a is alpha, b is beta, N is the sample size, Create matrix A

    A = np.zeros((N,N))
    row=np.zeros(N)
    row[0]=-2*a - 6*b
    row[1]=a+4*b
    row[2]=-b
    row[N-2]=-b
    row[N-1]=a+4*b
    
    for i in range(N):
        A[i] = np.roll(row, i)
    return A


# In[4]:


def xcheckBounds(x,img):
    x[ x < 0 ] = 0
    x[ x > img.shape[1]-1 ] = img.shape[1]-1
    x=x.round().astype(int)
    return x

def ycheckBounds(y,img):
    y[ y < 0 ] = 0
    y[ y > img.shape[0] ] = img.shape[0]-1
    y=y.round().astype(int)
    return y


# In[5]:


def snake(x, y, N, a, b, gx, gy, gamma, n_iters,img):
   
    A = createA(a,b,N)
    B = np.linalg.inv(np.identity(N) - gamma*A)
    snakes=[]

    for i in range(n_iters):
        if(i%5==0):
            plt.plot(x,y)
        
        fx=np.zeros(x.shape[0])
        fy=np.zeros(x.shape[0])
        for j in range(0, x.shape[0]):
            fx[j]=gx[y[j]][x[j]]
            fy[j]=gy[y[j]][x[j]]
        px = np.matmul(B, x + gamma*fx)
        px=xcheckBounds(px,img)
        py = np.matmul(B, y + gamma*fy)
        py=ycheckBounds(py,img)
        snakes.append( (py.copy(),py.copy()) )
        x, y = px.copy(), py.copy()
    return x,y


# In[6]:


drawing = False # true if mouse is pressed
ix,iy=-1,-1
cx=[]
cy=[]
# mouse callback function
def draw_circle(event,x,y,flags,param):
    global ix,iy,drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix,iy = x,y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.circle(img,(x,y),5,(0,0,255),-1)
            cx.append(x)
            cy.append(y)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
                


# In[7]:


filename="C:\\Users\\start\\Downloads\\as2\\p2.jpg"
#filename="C:\\Users\\start\\Downloads\\as2\\pc.jpg"
#filename="C:\\Users\\start\\Downloads\\image2.jpg"
outfile="C:\\Users\\start\\Downloads\\out.jpg"
textfile="C:\\Users\\start\\Downloads\\snake.txt"
#image could also be obtained from camera by calling CaptureVid()
#read in original image
image=cv2.imread(filename)
showImg(image)
cv2.waitKey(0)
cv2.destroyAllWindows()

#convert image to gray scale, adding noise and smoothing it

img1=image.copy()
img1=grayImg(img1)
img1=noise(img1)
smooth(img1)
img1=imgp
showImg(img1)

gx,gy=gradient(img1)
print(gx.shape)
img=img1.copy()

'''
#initial contour
t = np.arange(0, 2*np.pi, 0.1)
x = 600+320*np.cos(t)
x=xcheckBounds(x,image)
y = 320+190*np.sin(t)
y=ycheckBounds(y,image)
'''

#draw the initial contour with mouse
cx=[]
cy=[]
cv2.setMouseCallback('image',draw_circle)
while(1):
    cv2.imshow('image',img)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
cv2.destroyAllWindows()

showImg(img1)
while(1):
    k = cv2.waitKey(0)
    if k==ord('s'):
        #perform snake func
        img2=img1.copy()
        x=np.array(cx)
        y=np.array(cy)
        x=x.astype(int)
        y=y.astype(int)

        N=x.shape[0]

        #parameters
        alpha = 1
        beta = 0.001
        gamma = 100
        iterations = 100
        rx,ry=snake(x,y,N,alpha,beta,gx,gy,gamma,iterations,image)
        #print(rx,ry)
        #plt.plot(rx,ry)
        for i in range(0,rx.shape[0]-1):
            cv2.line(img2, (rx[i], ry[i]), (rx[i+1], ry[i+1]), (0, 0, 0))
        cv2.imshow("image",img2)
    elif k==ord('w'):
        #save image
        cv2.imwrite(outfile,img2)
    elif k==ord('c'):
        #show mouse contour
        img2=img.copy()
        showImg(img2)
    elif k==ord('o'):
        #show original gray image
        img2=img1.copy()
        showImg(img2)
    elif k==ord('h'):
        #print help menu
        f=open(textfile,"r")
        print(f.read()) 
        
    elif k==27:
        cv2.destroyAllWindows()
        break
    


# In[ ]:





# In[ ]:





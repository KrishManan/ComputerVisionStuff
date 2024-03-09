import cv2
import numpy as np

#importing image as grayscale
original=cv2.imread("images/Pythonlogo.png")
original=cv2.resize(original,(0,0),fx=0.5,fy=0.5)
originalgray=cv2.cvtColor(original,cv2.COLOR_BGR2GRAY)

#Roberts Cross Edge Detection
def Robertscross(image):
    #derivative kernels
    gx=np.array([[1,0,],
                [0,-1]])
    gy=np.array([[0,-1,],
                [1,0]])

    #applying partial derivatives 
    gradient_x=cv2.filter2D(src=image,ddepth=-1,kernel=gx)
    gradient_y=cv2.filter2D(src=image,ddepth=-1,kernel=gy)

    cv2.imshow("Roberts derivative in x",gradient_x)
    cv2.waitKey(1000)
    cv2.imshow("Roberts derivative in y",gradient_y)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()
    cv2.imwrite('images/Robertsgx.png', gradient_x)
    cv2.imwrite('images/Robertsgy.png', gradient_y)

    #finding the total magnitude

    magnitude=np.sqrt(gradient_x**2 + gradient_y**2)


    max=np.max(magnitude)
    magnitude=np.multiply(magnitude,(254/max))
    magnitude=np.array(magnitude,np.uint8)
    return magnitude

#Sobel Edge Detection
def Sobel(image):
    #derivative kernels
    gx=np.array([[-1,0,1],
                [-2,0,2],
                [-1,0,1]])
    gy=np.array([[1,2,1],
                [0,0,0],
                [-1,-2,-1]])

    #applying partial derivatives 
    gradient_x=cv2.filter2D(src=image,ddepth=-1,kernel=gx)
    gradient_y=cv2.filter2D(src=image,ddepth=-1,kernel=gy)

    cv2.imshow("Sobel derivative in x",gradient_x)
    cv2.waitKey(1000)
    cv2.imshow("Sobel derivative in y",gradient_y)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()

    #finding the total magnitude

    magnitude=np.sqrt(gradient_x**2 + gradient_y**2)


    max=np.max(magnitude)
    magnitude=np.multiply(magnitude,(254/max))
    magnitude=np.array(magnitude,np.uint8)
    return magnitude

#Prewitt Edge Detection
def Prewitt(image):
    #derivative kernels
    gx=np.array([[-1,0,1],
                [-1,0,1],
                [-1,0,1]])
    gy=np.array([[1,1,1],
                [0,0,0],
                [-1,-1,-1]])

    #applying partial derivatives 
    gradient_x=cv2.filter2D(src=image,ddepth=-1,kernel=gx)
    gradient_y=cv2.filter2D(src=image,ddepth=-1,kernel=gy)

    cv2.imshow("Prewitt derivative in x",gradient_x)
    cv2.waitKey(1000)
    cv2.imshow("Prewitt derivative in y",gradient_y)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()

    #finding the total magnitude

    magnitude=np.sqrt(gradient_x**2 + gradient_y**2)


    max=np.max(magnitude)
    magnitude=np.multiply(magnitude,(254/max))
    magnitude=np.array(magnitude,np.uint8)
    return magnitude

#Sobel Edge Detection
def Sobel(image):
    #derivative kernels
    gx=np.array([[-1,0,1],
                [-2,0,2],
                [-1,0,1]])
    gy=np.array([[1,2,1],
                [0,0,0],
                [-1,-2,-1]])

    #applying partial derivatives 
    gradient_x=cv2.filter2D(src=image,ddepth=-1,kernel=gx)
    gradient_y=cv2.filter2D(src=image,ddepth=-1,kernel=gy)

    cv2.imshow("Sobel derivative in x",gradient_x)
    cv2.waitKey(1000)
    cv2.imshow("Sobel derivative in y",gradient_y)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()

    #finding the total magnitude

    magnitude=np.sqrt(gradient_x**2 + gradient_y**2)


    max=np.max(magnitude)
    magnitude=np.multiply(magnitude,(254/max))
    magnitude=np.array(magnitude,np.uint8)
    return magnitude

#Larger Sobel (5x5) With Gaussian Edge Detection
def Largesobel(image):
    #derivative kernels
    gx=np.array([[-1,-2,0,2,1],
                 [-2,-3,0,3,2],
                 [-3,-5,0,5,3],
                 [-2,-3,0,3,2],
                 [-1,-2,0,2,1]
                ])
    gy=np.array([[1,2,3,2,1],
                 [2,3,5,3,2],
                 [0,0,0,0,0],
                 [2,3,5,3,2],
                 [1,2,3,2,1],])

    #applying partial derivatives 
    gradient_x=cv2.filter2D(src=image,ddepth=-1,kernel=gx)
    gradient_y=cv2.filter2D(src=image,ddepth=-1,kernel=gy)

    cv2.imshow("Large Sobel derivative in x",gradient_x)
    cv2.waitKey(1000)
    cv2.imshow("Large Sobel derivative in y",gradient_y)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()
    cv2.imwrite('images/LargeSobelgx.png', gradient_x)
    cv2.imwrite('images/LargeSobelgy.png', gradient_y)

    #finding the total magnitude

    magnitude=np.sqrt(gradient_x**2 + gradient_y**2)



    max=np.max(magnitude)
    magnitude=np.multiply(magnitude,(254/max))
    magnitude=np.array(magnitude,np.uint8)
    return magnitude


#Laplacian 
def Laplacian(image):
    #2 types of laplacian kernels
    laplacian1=np.array([[0,-1,0],
                         [-1,4,-1],
                         [0,-1,0]
                         ])
    laplacian2=np.array([[-1,-1,-1],
                         [-1,8,-1],
                         [-1,-1,-1]
                         ])

    #applying laplacian kernel
    lapimage1=cv2.filter2D(src=image,ddepth=-1,kernel=laplacian1)
    lapimage2=cv2.filter2D(src=image,ddepth=-1,kernel=laplacian2)

    cv2.imshow("Large Sobel derivative in x",lapimage1)
    cv2.waitKey(1000)
    cv2.imshow("Large Sobel derivative in y",lapimage2)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()

    return lapimage1,lapimage2

    

cv2.imshow("Original",original)
cv2.waitKey()
cv2.imshow("Original",originalgray)
cv2.waitKey()
cv2.destroyAllWindows()

cv2.imshow("Robertscross", Robertscross(originalgray))
cv2.waitKey()
cv2.imwrite("images/RobertsFinal.png",Robertscross(originalgray))
cv2.destroyAllWindows()

cv2.imshow("Prewitt", Prewitt(originalgray))
cv2.waitKey()
cv2.destroyAllWindows()

cv2.imshow("Sobel", Sobel(originalgray))
cv2.waitKey()
cv2.destroyAllWindows()

cv2.imshow("Sobel 5x5 with gaussian", Largesobel(originalgray))
cv2.waitKey()
cv2.imwrite("images/LargeSobelFinal.png",Largesobel(originalgray))
cv2.destroyAllWindows()

lapimage1,lapimage2=Laplacian(originalgray)
cv2.imshow("Laplacian1", lapimage1)
cv2.imshow("Laplacian2", lapimage2)
cv2.waitKey()
cv2.imwrite('images/Laplacian1.png',lapimage1)
cv2.imwrite('images/Laplacian2.png',lapimage2)
cv2.destroyAllWindows()
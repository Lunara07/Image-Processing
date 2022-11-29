import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
from sklearn.metrics import mean_squared_error

#iterating through contours and labels and if contour is within the region of any rectangle label then increase number of true positive value
def validation(img_val1, img_l, cnt):
    person=0 #true positive value
    found=False
    img_val = img_val1.copy()
    for c in cnt:
        for index, row in img_l.iterrows():
            found = False
            x_r = row['bbox_x'] + row['bbox_width']
            y_r = row['bbox_y'] + row['bbox_height']
            for cn in c:
                if (cn[0][0] in range(row['bbox_x'], x_r+1)) and (cn[0][1] in range(row['bbox_y'], y_r+1)):
                    found = True
                else: 
                    found = False
            if (found):
                person+=1
                (x, y, w, h) = cv2.boundingRect(c)
                img_val = cv2.rectangle(img_val1, (x, y), (x + w, y + h), (0, 0, 255), 2)
    return person, img_val1


if __name__ == "__main__":
    #read arguments
    img_n = sys.argv[1] #img name
    csv = sys.argv[2] #csv file name
    
    #reading the image
    img_o = cv2.imread("C:/Users/35841/Desktop/Masters/Image Processing/project/"+img_n+".jpg", cv2.IMREAD_COLOR) #original image
    grayscale_img = cv2.imread("C:/Users/35841/Desktop/Masters/Image Processing/project/"+img_n+".jpg", cv2.IMREAD_GRAYSCALE) #grayscale image
    
    #blurring
    blur =cv2.GaussianBlur(grayscale_img,(11,11),0)
    img_edges = blur.copy()

    #canny edge detection for different images based on the sunlight direction
    if (img_n=='img_2' or img_n=='img_4' or img_n=='img_5' or img_n=='img_6'):
        img_edges = cv2.Canny(blur, threshold1=30, threshold2=120, apertureSize=3)
    else:
        img_edges = cv2.Canny(blur, threshold1=30, threshold2=115, apertureSize=3)

    #dilation
    se = np.ones((1, 1), dtype='uint8') #structuring element
    dilated =cv2.dilate(img_edges,se, iterations =2)
    
    #finding contours
    (cnt, heirarchy ) = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #filtering out unnecessary contours  
    cnt = [c for c in cnt if c[0][0][1] > 450] #elimination of trees and ships
    cnt = [c for c in cnt if cv2.boundingRect(c)[2]*cv2.boundingRect(c)[3]<600] #elimination of contours of large objects
    cnt = [c for c in cnt if (c[0][0][1] < 900 or cv2.boundingRect(c)[2]*cv2.boundingRect(c)[3]>10)] #elimination of the small contours in the lower part of the image
    cnt = [c for c in cnt if (c[0][0][0] < 1830 or c[0][0][1] < 750)] #elimination of contours of regions with no people being expected

    #drawing contours
    rgb=img_o.copy()
    rgb =cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    cv2.drawContours(rgb, cnt, -1, (0,255,0),2)
    
    #reading the csv file
    img_l = pd.read_csv('C:/Users/35841/Desktop/Masters/Image Processing/project/'+csv+'.csv')
    #validation
    img_val1 = rgb.copy()
    person, img_val = validation(img_val1, img_l,cnt)
    print('Predicted number of people:', len(cnt))
    print('Manual annotation:', img_l.shape[0])   # It outputs the object count
    print("True positive values:", person)
    # Display image
    fig, axs = plt.subplots(3, 2)
    # Show one image per subplot
    axs[0, 0].set_title('Original')
    axs[0, 0].imshow(img_o)
    axs[0, 1].set_title('Blur')
    axs[0, 1].imshow(blur, cmap='gray')
    axs[1, 0].set_title('Edge')
    axs[1, 0].imshow(img_edges, cmap='gray')
    axs[1, 1].set_title('Dilated')
    axs[1, 1].imshow(dilated, cmap='gray')
    axs[2, 0].set_title('Final')
    axs[2, 0].imshow(rgb)
    axs[2, 1].set_title('Validation')
    axs[2, 1].imshow(img_val)
    # Display figure
    plt.show()

    #save the result and validation
    cv2.imwrite("C:/Users/35841/Desktop/Masters/Image Processing/project/"+img_n+"_result.jpg", rgb)
    cv2.imwrite("C:/Users/35841/Desktop/Masters/Image Processing/project/"+img_n+"_validation.jpg", img_val)

    #calculation of metrics for all images
    true_pos = [0,5,17,72,113,118,141,127,177,138] 
    pred = [1,13,27,113,176,240,259,249,255,262]
    true = [0,12,37,110,152,226,232,236,228,238]
    print('Mean squarred error', mean_squared_error(true,pred))
    accuracy = 0
    for i in range(1,len(true)):
        accuracy+=true_pos[i]/true[i]
    accuracy = accuracy/len(true)    
    print('Average accuracy is:', accuracy)

    

    

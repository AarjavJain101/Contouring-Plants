import cv2 as cv
import numpy as np
import os
import time
import shutil

# Runs probably 9.5/10 for Bean in 1 min 40 sec.

folder_path = "C:/Users/Aarja/Documents/UBC/pythonProject/OpenCV_Learning/Bean"
scale = 0.2  # 0.4 for Maize, 0.2 for Bean, 0.2 for Maize2

# Function to do the Processing
def doProcessing(frame, scale):
    # First we rescale the image
    img_rescale = rescale(frame, scale)

    # Preprocessing Stage

    # First we bring out the green in the image
    b, g, r = cv.split(img_rescale)
    for i in range(b.shape[0]):
        for j in range(b.shape[1]):
            if g[i, j] > b[i, j] and g[i, j] > r[i, j]:
                g[i, j] = 255
            else:
                g[i, j] = 0
    merged = cv.merge([b, g, r])

    # Now we grayscale and remove noise
    gray = cv.cvtColor(merged, cv.COLOR_BGR2GRAY)
    median = cv.medianBlur(gray, 7)
    for i in range(5):
        median = cv.medianBlur(median, 7)

    # Start contour processing
    ret, thresh = cv.threshold(median, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    contours, hierarchies = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

    # Check if there are way too many contours (probably means that there are no plants)
    if len(contours) >= 130:
        dimensions = (frame.shape[1], frame.shape[0])
        img_rescale = cv.resize(img_rescale, dimensions, interpolation=cv.INTER_AREA)
        save_image_as_jpg(img_rescale)
    else:
        # This function returns the modified threshold image with morphological Open
        new_Thresh = getNewThresh(thresh)

        contours, hierarchies = cv.findContours(new_Thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv.contourArea, reverse=True)

        print(f'{len(contours)} contour(s) found!')

        # Eliminate possibly weed like contours 
        minArea = cv.contourArea(contours[getMinArea(contours)])
        rmvIndex = [0 for i in range(len(contours))]
        index = 0;
        for i in range(len(contours)):
            if cv.contourArea(contours[i]) <= minArea:
                rmvIndex[index] = i
                index = index + 1
        for i in range(index):
            del contours[rmvIndex[i] - i]

        # Display the contours and the image
        if len(contours) == 0:
            img_rescale = rescale(img_rescale, 1.5)
            # Scale image back up for visibility and save into 'Converted' folder
            dimensions = (frame.shape[1], frame.shape[0])
            img_rescale = cv.resize(img_rescale, dimensions, interpolation=cv.INTER_AREA)
            save_image_as_jpg(img_rescale)
        else:
            for i in range(len(contours)):
                cv.drawContours(img_rescale, contours, i, (255, 255, 255), 2)

            img_rescale = rescale(img_rescale, 1.5)
            # Scale image back up for visibility and save into 'Converted' folder
            dimensions = (frame.shape[1], frame.shape[0])
            img_rescale = cv.resize(img_rescale, dimensions, interpolation=cv.INTER_AREA)
            save_image_as_jpg(img_rescale)

def save_image_as_jpg(image, output_dir="C:/Users/Aarja/Documents/UBC/pythonProject/OpenCV_Learning/Converted"):
    # Make random filename
    timestamp = str(int(round(time.time() * 1000)))
    output_file = os.path.join(output_dir, timestamp + ".jpg")

    # Save the image as a JPG file
    cv.imwrite(output_file, image)

def getMinArea(contours):
    # Create an array to find the ratio of areas of consecutive contours
    difference = [0 for i in range(len(contours) - 1)]
    # First make sure all contours are big enough so that they do not give divide by 0 errors
    for i in range(len(contours)):
        if cv.contourArea(contours[i]) < 20:
            contours[i] = contours[i-1]
    for i in range(len(contours) - 1):
        difference[i] = cv.contourArea(contours[i]) / cv.contourArea(contours[i + 1])
    if len(difference) == 0:
        return 0
    # To return the index of the first big ratio spike we check if the area at the spike is > 4000 to be sure
    for i in range(len(contours) - np.argmax(difference)):
        if cv.contourArea(contours[np.argmax(difference) + i]) > 4000:
            continue
        else:
            return np.argmax(difference) + i
    return np.argmax(difference)
            

def rescale(image, scale):
    width = int(image.shape[1] * scale)
    height = int(image.shape[0] * scale)
    dimensions = (width, height)
    return cv.resize(image, dimensions, interpolation=cv.INTER_AREA)


def getRowOrCol(thresh, rowOrCol):
    # First index in array is the row or column, the second index is the spacing
    returnVals = [0 for i in range(2)]
    averageCol = [0 for i in range(thresh.shape[1])]
    # Loop to get average pixel value for each column
    for i in range(thresh.shape[1]):
        sum = 0
        for j in range(thresh.shape[0]):
            sum = sum + thresh[j, i]
        averageCol[i] = sum/thresh.shape[0]
    
    # Loop to get the average pixel value for each row
    averageRow = [0 for i in range(thresh.shape[0])]
    for i in range(thresh.shape[0]):
        sum = 0
        for j in range(thresh.shape[1]):
            sum = sum + thresh[i, j]
        averageRow[i] = sum/thresh.shape[1]
    
    # Now determine if the plants are in a row or in a column
    index = 0
    if max(averageRow) >= max(averageCol):
        index = np.argmax(averageRow)
        if rowOrCol == "row":
            # Figure out Row Spacing
            rowUp = 0
            for i in range(np.argmax(averageRow)):
                if averageRow[np.argmax(averageRow) - i] > 0.1*max(averageRow):
                    continue
                else:
                    rowUp = np.argmax(averageRow) - i
                    break

            rowDown = 0
            for i in range(thresh.shape[0] - np.argmax(averageRow) - 1):
                if averageRow[np.argmax(averageRow) + i] > 0.1*max(averageRow):
                    continue
                else:
                    rowDown = np.argmax(averageRow) + i
                    break
            
            spacingRow = rowDown - rowUp

            returnVals[0] = index
            returnVals[1] = spacingRow/(2 * thresh.shape[0])
            return returnVals
        elif rowOrCol == "col":
            return None
    else:
        index = np.argmax(averageCol)
        if rowOrCol == "row":
            return None
        elif rowOrCol == "col":
            # Figure out Column Spacing
            colRight = 0
            for i in range(thresh.shape[1] - np.argmax(averageCol) - 1):
                if averageCol[np.argmax(averageCol) + i] > 0.25*max(averageCol):
                    continue
                else:
                    colRight = np.argmax(averageCol) + i
                    break

            colLeft = 0
            for i in range(np.argmax(averageCol)):
                if averageCol[np.argmax(averageCol) - i] > 0.25*max(averageCol):
                    continue
                else:
                    colLeft = np.argmax(averageCol) - i
                    break


            spacingCol = colRight - colLeft

            returnVals[0] = index
            returnVals[1] = spacingCol/(2 * thresh.shape[1])
            return returnVals


def morphOutsideCols(thresh, column_start, column_end):
    height, width = thresh.shape

    # Split the image along the specified column numbers
    left_half = thresh[:, :column_start]
    middle_part = thresh[:, column_start:column_end]
    right_half = thresh[:, column_end:]

    # Apply morphologyEx to everything outside the columns
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    left_half = cv.morphologyEx(left_half, cv.MORPH_OPEN, kernel, iterations=10)
    right_half = cv.morphologyEx(right_half, cv.MORPH_OPEN, kernel, iterations=10)

    # Recombine the parts to form the full image
    result = cv.hconcat([left_half, middle_part, right_half])
    return result

def morphOutsideRows(thresh, row_start, row_end):
    height, width = thresh.shape

    # Split the image along the specified row numbers
    top_half = thresh[:row_start, :]
    middle_part = thresh[row_start:row_end, :]
    bottom_half = thresh[row_end:, :]

    # Apply morphologyEx to everything outside the rows
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    top_half = cv.morphologyEx(top_half, cv.MORPH_OPEN, kernel, iterations=9)
    bottom_half = cv.morphologyEx(bottom_half, cv.MORPH_OPEN, kernel, iterations=9)

    # Recombine the parts to form the full image
    result = cv.vconcat([top_half, middle_part, bottom_half])
    return result

def getNewThresh(thresh):
    # Get the row or the column of greatest average (depends on image)
    column = getRowOrCol(thresh, "col")
    row = getRowOrCol(thresh, "row")

    if column != None:
        # Create a column space by going equally outwards from the specified column
        colLeft = int(max(column[0] - column[1]*thresh.shape[1], 0))
        colRight = int(min(column[0] + column[1]*thresh.shape[1], thresh.shape[1]) - 1)

        temp = colRight
        if colLeft > colRight:
            colRight = colLeft
            colLeft = temp
        
        # Hard filter outside the columns
        new_thresh = morphOutsideCols(thresh, colLeft, colRight)
        column = getRowOrCol(new_thresh, "col")
        colLeft = int(max(column[0] - 0.2*thresh.shape[1], 0))
        colRight = int(min(column[0] + 0.2*thresh.shape[1], thresh.shape[1]))

        # Remove everything outside a large enough row space (kills stray weeds)
        return eraseOutsideCols(colLeft, colRight, new_thresh)


    if row != None:
        # Create a row space by going equally outwards from the specified row
        rowUp = int(max(row[0] - abs(row[1]*thresh.shape[0]), 0))
        rowDown = int(min(row[0] + abs(row[1]*thresh.shape[0]), thresh.shape[0]) - 1)

        temp = rowDown
        if rowUp > rowDown:
            rowDown = rowUp
            rowUp = temp

        # Hard filter outside the columns
        new_thresh = morphOutsideRows(thresh, rowUp, rowDown)
        row = row
        rowUp = int(max(row[0] - 0.35*thresh.shape[0], 0))
        rowDown = int(min(row[0] + 0.35*thresh.shape[0], thresh.shape[0]))
        
        # Remove everything outside a large enough row space (kills stray weeds)
        return eraseOutsideRows(rowUp, rowDown, new_thresh)
    

def eraseOutsideCols(left, right, g):
    for i in range(g.shape[0]):
        for j in range(g.shape[1]):
            if j < left or j > right:
                g[i, j] = 0
    return g

def eraseOutsideRows(top, bottom, g):
    for i in range(g.shape[0]):
        for j in range(g.shape[1]):
            if i < top or i > bottom:
                g[i, j] = 0
    return g


output_dir = "C:/Users/Aarja/Documents/UBC/pythonProject/OpenCV_Learning/Converted"
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)

# Create the output directory
os.makedirs(output_dir)

for filename in os.listdir(folder_path):
    if filename.endswith(".jpg"):
        image_path = os.path.join(folder_path, filename)
        image = cv.imread(image_path)
        doProcessing(image, scale)
    else:
        continue

cv.waitKey(0)
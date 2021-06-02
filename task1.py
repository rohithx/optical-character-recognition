"""
Character Detection

The goal of this task is to implement an optical character recognition system consisting of Enrollment, Detection and Recognition sub tasks

Please complete all the functions that are labelled with '# TODO'. When implementing the functions,
comment the lines 'raise NotImplementedError' instead of deleting them.

Do NOT modify the code provided.
Please follow the guidelines mentioned in the project1.pdf
Do NOT import any library (function, module, etc.).
"""


import argparse
import json
import os
import glob
import cv2
import numpy as np


def read_image(img_path, show=False):
    """Reads an image into memory as a grayscale array.
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if show:
        show_image(img)

    return img

def show_image(img, delay=1000):
    """Shows an image.
    """
    cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('image', img)
    cv2.waitKey(delay)
    cv2.destroyAllWindows()

def parse_args():
    parser = argparse.ArgumentParser(description="cse 473/573 project 1.")
    parser.add_argument(
        "--test_img", type=str, default="./data/test_img.jpg",
        help="path to the image used for character detection (do not change this arg)")
    parser.add_argument(
        "--character_folder_path", type=str, default="./data/characters",
        help="path to the characters folder")
    parser.add_argument(
        "--result_saving_directory", dest="rs_directory", type=str, default="./",
        help="directory to which results are saved (do not change this arg)")
    args = parser.parse_args()
    return args

def ocr(test_img, characters):
    """Step 1 : Enroll a set of characters. Also, you may store features in an intermediate file.
       Step 2 : Use connected component labeling to detect various characters in an test_img.
       Step 3 : Taking each of the character detected from previous step,
         and your features for each of the enrolled characters, you are required to a recognition or matching.

    Args:
        test_img : image that contains character to be detected.
        characters_list: list of characters along with name for each character.

    Returns:
    a nested list, where each element is a dictionary with {"bbox" : (x(int), y (int), w (int), h (int)), "name" : (string)},
        x: row that the character appears (starts from 0).
        y: column that the character appears (starts from 0).
        w: width of the detected character.
        h: height of the detected character.
        name: name of character provided or "UNKNOWN".
        Note : the order of detected characters should follow english text reading pattern, i.e.,
            list should start from top left, then move from left to right. After finishing the first line, go to the next line and continue.
        
    """
    # TODO Add your code here. Do not modify the return and input arguments

    enroll = enrollment(characters)

    the_list = detection(test_img)
    
    final = recognition(enroll, the_list, test_img)

    return final   

    #raise NotImplementedError

def enrollment(characters):
    """ Args:
        You are free to decide the input arguments.
    Returns:
    You are free to decide the return.
    """
    # TODO: Step 1 : Your Enrollment code should go here.
    
    def xroll(img, thresh = 127):
        no_row = len(img)
        no_col = len(img[0])
        x = None
        y = None
        hx = None
        hy = None
        for j in range(0,no_row):
            for k in range(0,no_col):
                if img[j,k] <= thresh:
                    if y == None and x == None:
                        y = j
                        hy = j
                        x = k
                        hx = k
                    else:
                        if y > j:
                            y = j
                        if hy < j:
                            hy = j
                        if x > k:
                            x = k
                        if hx < k:
                            hx = k
        return img[y-1:hy+2,x-1:hx+2]
    
    def tnb(img,thres = 127):
        t, bimg = cv2.threshold(img,thres,255,cv2.THRESH_BINARY)
        return bimg
    
    def rs(img, size = (64,64)):
        pimg = cv2.resize(img,size)
        return pimg
        
    def do(img):
        i = tnb(img)
        j = rs(i)
        k = tnb(j)
        return k
    
    def doit(img):
        roll = []
        for i in range(0,4):
            for j in range(0,4):
                tmg = img[16*i:16*(i+1),16*j:16*(j+1)]
                roll.append((tmg==0).sum())   
        return roll
    
    preroll = []
    for i in characters:
        nmg = xroll(i[1])
        bimg = do(nmg)
        feat = doit(bimg)
        preroll.append({'char':i[0],'feat':feat})      
    
    return preroll      

def detection(test_img):
    """ 
    Use connected component labeling to detect various characters in an test_img.
    Args:
        You are free to decide the input arguments.
    Returns:
    You are free to decide the return.
    """
    # TODO: Step 2 : Your Detection code should go here.
    no_row = len(test_img)
    no_col = len(test_img[0])
    thresh = 100
    label_img = np.zeros((no_row, no_col))
    uf = []
    
    #first pass
    l = 1
    for i in range(0, no_row):
        for j in range(0, no_col):
            if test_img[i,j] < 255 - thresh:
                if i == 0 and j == 0:
                    label_img[i,j] = l
                    l = l+1
                elif i == 0 and j != 0:
                    if label_img[i,j-1] == 0:
                        label_img[i,j] = l
                        l = l+1
                    else:
                        label_img[i,j] = label_img[i,j-1]
                elif i != 0 and j == 0:
                    if label_img[i-1,j] == 0:
                        label_img[i,j] = l
                        l = l+1
                    else:
                        label_img[i,j] = label_img[i-1,j]
                else:
                    if label_img[i-1,j] == 0 and label_img[i,j-1] == 0:
                        label_img[i,j] = l
                        l = l+1
                    elif label_img[i-1,j] == 0 and label_img[i,j-1] != 0:
                        label_img[i,j] = label_img[i,j-1]
                    elif label_img[i-1,j] != 0 and label_img[i,j-1] == 0:
                        label_img[i,j] = label_img[i-1,j]
                    else:
                        if label_img[i,j-1] == label_img[i-1,j]:
                            label_img[i,j] = label_img[i,j-1]
                        else:
                            label_img[i,j] = min(label_img[i-1,j],label_img[i,j-1])
                            uf.append([min(label_img[i,j-1],label_img[i-1,j]),max(label_img[i,j-1],label_img[i-1,j])])
    l = l - 1
    
    #2nd pass
    def ufds(x,l):
        b = []
        for i in range(1,l+1):
            b.append([i])
        for j in x:
            i1 = 0
            i2 = 0
            for k in range(0,len(b)):
                if j[0] in b[k]:
                    i1 = k
                if j[1] in b[k]:
                    i2 = k
            if i1 != i2:
                b[i1] = b[i1] + b[i2]
                del b[i2]
        return b
    
    bman = ufds(uf,l)
    
    #3rd pass
    uf_arr = np.zeros(l)
    for i in range(0,len(uf_arr)):
        for j in bman:
            if i+1 in j:
                uf_arr[i] = min(j) 
    
    fin_img = np.zeros((no_row, no_col))
    for i in range(0, no_row):
        for j in range(0, no_col):
            if label_img[i,j] != 0:
                fin_img[i,j] = uf_arr[int(label_img[i,j] - 1)]
    
    all_label = []
    '''
    for i in bman:
        all_label.append(min(i))
    '''       
    for i in range(0, no_row):
        for j in range(0, no_col):
            if fin_img[i,j] !=0 and fin_img[i,j] not in all_label:
                all_label.append(fin_img[i,j])
    
    # main image
    k_img = np.zeros((no_row, no_col))
    for i in range(0, no_row):
        for j in range(0, no_col):
            if fin_img[i,j] != 0:
                k_img[i,j] = (all_label.index(fin_img[i,j]) + 1)
    
    k_list = []
    for i in range(1,len(all_label)+1):
        x = None
        y = None
        hx = None
        hy = None
        for j in range(0,no_row):
            for k in range(0,no_col):
                if k_img[j,k] == i:
                    if y == None and x == None:
                        y = j
                        hy = j
                        x = k
                        hx = k
                    else:
                        if y > j:
                            y = j
                        if hy < j:
                            hy = j
                        if x > k:
                            x = k
                        if hx < k:
                            hx = k
        k_list.append({"bbox": [x, y, hx-x, hy-y]})
    
    return k_list
    
    #raise NotImplementedError

def recognition(enroll, the_list, test_img):
    """ 
    Args:
        You are free to decide the input arguments.
    Returns:
    You are free to decide the return.
    """
    # TODO: Step 3 : Your Recognition code should go here.
    def xroll(img, thresh = 127):
        no_row = len(img)
        no_col = len(img[0])
        x = None
        y = None
        hx = None
        hy = None
        for j in range(0,no_row):
            for k in range(0,no_col):
                if img[j,k] <= thresh:
                    if y == None and x == None:
                        y = j
                        hy = j
                        x = k
                        hx = k
                    else:
                        if y > j:
                            y = j
                        if hy < j:
                            hy = j
                        if x > k:
                            x = k
                        if hx < k:
                            hx = k
        return img[y-1:hy+2,x-1:hx+2]
    
    def tnb(img,thres = 127):
        t, bimg = cv2.threshold(img,thres,255,cv2.THRESH_BINARY)
        return bimg
    
    def rs(img, size = (64,64)):
        pimg = cv2.resize(img,size)
        return pimg
        
    def do(img):
        i = tnb(img)
        j = rs(i)
        k = tnb(j)
        return k
    
    def doit(img):
        roll = []
        for i in range(0,4):
            for j in range(0,4):
                tmg = img[16*i:16*(i+1),16*j:16*(j+1)]
                roll.append((tmg==0).sum())   
        return roll
    
    the_list_err = []
    for i in the_list:
        timg = test_img[i['bbox'][1]-1:(i['bbox'][1]+i['bbox'][3]+2),i['bbox'][0]-1:(i['bbox'][0]+i['bbox'][2]+2)] 
        nimg = xroll(timg)
        done = do(nimg)
        f = doit(done)
        err = []
        for k in enroll:
            t = 0
            for p in range(0,len(f)):
                t += ((f[p] - k['feat'][p])**2)
            err.append(t)
        the_list_err.append(err)
    
    good_list_index = []
    good_list = []
    for i in the_list_err:
        for j in i:
            if j < 10000:
                good_list_index.append(the_list_err.index(i))
                good_list.append(i.index(j))
    
    new_list = []
    for i in range(0,len(the_list)):
        new_list.append(the_list[i])
        if i in good_list_index:
            ind = good_list_index.index(i)
            p = good_list[ind]
            new_list[i]["name"] = enroll[p]['char']
        else:
            new_list[i]["name"] = "UNKNOWN"
    
    return new_list
    #raise NotImplementedError


def save_results(coordinates, rs_directory):
    """
    Donot modify this code
    """
    results = coordinates
    with open(os.path.join(rs_directory, 'results.json'), "w") as file:
        json.dump(results, file)


def main():
    args = parse_args()
    
    characters = []

    all_character_imgs = glob.glob(args.character_folder_path+ "/*")
    
    for each_character in all_character_imgs :
        character_name = "{}".format(os.path.split(each_character)[-1].split('.')[0])
        characters.append([character_name, read_image(each_character, show=False)])

    test_img = read_image(args.test_img)

    results = ocr(test_img, characters)

    save_results(results, args.rs_directory)


if __name__ == "__main__":
    main()

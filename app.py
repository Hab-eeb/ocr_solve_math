import uvicorn
from fastapi import FastAPI, File, UploadFile
import numpy as np
import pandas as pd 
import cv2
import os
import io
from PIL import Image
from imutils import contours
from keras.models import model_from_json

app = FastAPI()
def dig_contour(img):
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY )
    ret, thresh = cv2.threshold(img_gray, 150,255, cv2.THRESH_BINARY)
    conts , hierarchy = cv2.findContours(image=thresh, mode= cv2.RETR_TREE, method = cv2.CHAIN_APPROX_SIMPLE)
    
    if len(conts) < 1 :
        return "There are no more lines of numbers "
    digit_contours = [] 
    
    for c in conts:
        (x,y,w,h) = cv2.boundingRect(c)
        
        if (w >= 7 and x <= 120) and (h >= 30 and h <= 120):
            digit_contours.append(c)
    if len(digit_contours) < 1:
        return "There are no more lines of numbers "
    digit_contours = contours.sort_contours(digit_contours, method="left-to-right")[0]
    
    return digit_contours

def line_contour(img):
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY )
    ret, thresh = cv2.threshold(img_gray, 150,255, cv2.THRESH_BINARY)
    conts , hierarchy = cv2.findContours(image=thresh, mode= cv2.RETR_TREE, method = cv2.CHAIN_APPROX_SIMPLE)
    
    if len(conts) < 1 :
        return "There are no more lines of numbers "
    digit_contours = [] 
    
    for c in conts:
        (x,y,w,h) = cv2.boundingRect(c)
        
        if (w >= 7 and w <= 120) and (h >= 30 and h <= 120):
            digit_contours.append(c)
    if len(digit_contours) < 1:
        return "There are no more lines of numbers "
    digit_contours = contours.sort_contours(digit_contours, method="top-to-bottom")[0]
    
    return digit_contours

def split_lines(img):

    lines_imgs = []
    
    contour_digits = line_contour(img)
    # print("done1")
    new_image, new_line = line_img(contour_digits,img)
    lines_imgs.append(new_line)
    contour_digits = line_contour(new_image)
    # print("done2")
    while type(contour_digits) == tuple:
        new_image, new_line = line_img(contour_digits,new_image) 
        if type(dig_contour(new_line)) == tuple:
          lines_imgs.append(new_line)
        # t=new_image
        if len(new_image) == 0 :
          break 
        contour_digits = line_contour(new_image)
    return lines_imgs

def line_img(contour_digits,img):
    
    (x,y,w,h) = cv2.boundingRect(contour_digits[0])
    
    n_i = img[y + h +140 :, :]
    n_l = img[ :y + h+ 170, :]
    
    return n_i, n_l 

def overlap(l1,l2,r1,r2):
    if (l1["x"] == r1["x"] or l1["y"] == r1["y"] or l2["x"] == r2["x"] or l2["y"] == r2["y"]):
        return False
    if (l1["x"] > r2["x"] or l2["x"] > r1["x"]):
        return False 
    if (l2["y"] > r1["y"] or l1["y"] > r2["y"]):
        return False 
    return True 

def remove_overlap(digs):
  new_digits =[]
  dig = list(digs)
  for ty in range(0,len(dig)-1):
    (x,y,w,h) = cv2.boundingRect(dig[ty]) 
    # roi_l = lines[0][y:y + h, x:x + w]
    l1 = {"x":x,"y":y,"w":w,"h":h}
    r1 = {"x":x + w,"y":y + h}
    
    (x,y,w,h) = cv2.boundingRect(dig[ty+1]) 
    # roi_l_1 = lines[0][y:y + h, x:x + w]
    l2 = {"x":x,"y":y,"w":w,"h":h}
    r2 = {"x":x + w,"y":y + h}
  
    if overlap(l1,l2,r1,r2):
      if (l1["w"] * l1["h"]) > l2["w"] * l2["h"] :
        # del dig[ty+1]
        new_digits.append(dig[ty+1])
      else:
        # del dig[ty]
        new_digits.append(dig[ty])
      # new_digits.append(digs[ty+1])
  for i in new_digits:
    dig.remove(i)

  return dig

# correct_eval = 56 # calculated from input gotten from user 
sign_keys = {0:"0",1:"1",2:"2",3:"3",4:"4",5:"5",6:"6",7:"7",8:"8",9:"9",11:"*",12:"-",13:"+"}
def eval_image(im):
  line_eval =[]
  lines = split_lines(im)
  for i in range(0,len(lines)):
    digits = dig_contour(lines[i])
    digits = remove_overlap(digits)
    num=""
    for l in digits:
      (x,y,w,h) = cv2.boundingRect(l)
      roi_l = lines[i][y:y + h, x:x + w] 

      img_num = cv2.resize(roi_l,(28,28))
      img_gray = cv2.cvtColor(img_num,cv2.COLOR_BGR2GRAY )
      # ret, thresh = cv2.threshold(img_gray, 150,255, cv2.THRESH_BINARY)
      im_resize = cv2.bitwise_not(img_gray)
      img_num = np.reshape(im_resize,(1,1,28,28)) 

      pred = loaded_model.predict(img_num)
      y_pred=np.argmax(pred,axis=1)
      # y_pred = [sign_keys[i]
      num = num + sign_keys[y_pred[0]]

    # print(num)
    # print()
    line_eval.append(num)
  # print(f"There are {len(line_eval)} lines " )
  t = f"There are {len(line_eval)} lines " 
  for e in range(0,len(line_eval)): 
    # print(f"The {e} line is :{line_eval[e]}")
    
    t = t + f" The line {e} has {len(line_eval[e])} characters and is approximately :{line_eval[e]}  "
  
  # s = print(t)
  
  
  return t
    # if correct_eval == eval(line_eval[e]):
    #   print(f"Line {e} is correct")
    # else:
    #   print(f"Line {e} is not correct")

json_file = open('n_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("n_model.h5")
# image = cv2.imread("image_6.jpg")
def load_image_into_numpy_array(data):
    return np.array(Image.open(io.BytesIO(data)))

@app.post("/")
async def get_some(file: UploadFile = File(...)):
    image = load_image_into_numpy_array(await file.read())

    # contents = io.BytesIO(file.read())
    # # nparr = np.fromstring(contents, np.uint8)
    # file_bytes = np.asarray(bytearray(contents.read()), dtype=np.uint8)
    # img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    # img = file.getvalue()
    # image = file.read() #cv2.imread(file.filename) 
    # file.close()

    ans = eval_image(image)

    # new_num = np.square(num)
    # print(f"Square of {num} is {new_num} ")



    return ans

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
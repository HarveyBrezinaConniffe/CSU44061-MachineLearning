import numpy as np
from PIL import Image

def applyKernel(arr, kernel):
  output = 0
  for row in range(len(arr)):
    for col in range(len(arr[0])):
      output += arr[row][col]*kernel[row][col]
  return output

def convolve(arr, kernel):
  img_h, img_w = arr.shape
  kernel_h, kernel_w = kernel.shape

  last_col = img_w-kernel_w
  last_row = img_h-kernel_h

  output = []

  for row in range(0, last_row):
    output.append([])
    for col in range(0, last_col):
      arr_section = arr[row:row+kernel_h, col:col+kernel_w]
      output[-1].append(applyKernel(arr_section, kernel)) 
  
  return output

im = Image.open("TCDHiking.jpg")
rgb = np.array(im.convert("RGB"))
r = rgb[:, :, 0]
Image.fromarray(np.uint8(r)).show()

kernel_1 = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
kernel_2 = np.array([[0, -1, 0], [-1, 8, -1], [0, -1, 0]])

convolved_1 = convolve(r, kernel_1)
convolved_2 = convolve(r, kernel_2)

Image.fromarray(np.uint8(np.array(convolved_1))).show()
Image.fromarray(np.uint8(np.array(convolved_2))).show()

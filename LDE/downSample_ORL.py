# import cv2
# idx = 0
# for person in range(40):
#     for pose in range(10):
#         img = cv2.imread("ORL/s"+str(person+1)+"/"+str(pose+1)+".pgm", flags=0) # 0: 读灰度图  type(img) = 'numpy.ndarray'
#         idx += 1
#         save_dir = "./ORL_23_28/orl"+str(idx)+".pgm"
#         cv2.imwrite(save_dir, img)

from PIL import Image
import torch.nn.functional
import torchvision.transforms.functional
import numpy as np
import matplotlib.pyplot as plt
import cv2

impath = r'E:\\文献\\科研训练\\4_NPE_LPP\\code_NPE\\att_faces\\'
orl_mode ="area"
idx = 0
for person in range(40):
    for pose in range(10):
        img = Image.open(impath+'s'+str(person+1)+'\\'+str(pose+1)+'.pgm')
        img = torchvision.transforms.functional.to_tensor(img) # 'PIL.JpegImagePlugin.JpegImageFile' -> 'torch.Tensor' 灰度: [1,H,W]
        img = img.unsqueeze(0) # [c,H,W] -> [1,c,H,W]

        img14 = torch.nn.functional.interpolate(img, scale_factor=0.25, mode=orl_mode) # [B,C,H,W]: batch_size * channel * H*W
        img14 = img14.squeeze(0) # [1,1,H,W] -> [1,H,W]
        img14 = img14.squeeze(0) # [1,H,W] -> [H,W]
        img14 = img14 * 255
        idx += 1
        save_dir = "./ORL_14_"+orl_mode+"/orl"+str(idx)+".pgm"
        cv2.imwrite(save_dir, np.array(img14))
# plt.figure()
# if img14.shape[0]==1: # 灰度图
#     img14 = img14.squeeze(0) # [1,H,W] -> [H,W]
#     plt.imshow(img14, cmap='gray') # 
# else:
#     img14 = img14.permute(1,2,0) # [c,H,W] -> [H,W,c]
#     plt.imshow(img14)
# plt.show()
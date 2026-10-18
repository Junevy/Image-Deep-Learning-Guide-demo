import torch.utils.tensorboard as tb
import cv2 as cv

img_path = r'D:\Documents\DEV\Python\datasets\steel\casting_data\train\def_front\cast_def_0_0.jpeg'
img1_path = r'D:\Documents\DEV\Python\datasets\steel\casting_data\train\def_front\cast_def_0_2.jpeg'

img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
img1 = cv.imread(img1_path, cv.IMREAD_GRAYSCALE)
# cv.imshow('img', img)
# cv.waitKey(0)
# cv.destroyAllWindows()
# print(img.shape)
# print(img.dtype)
# print(type(img))

writer = tb.SummaryWriter('tensor_board_test')
writer.add_image('img', img, dataformats='HW')
writer.add_image('img', img1, dataformats='HW')

for x in range(100):
    writer.add_scalar('loss', x, x)
writer.close()


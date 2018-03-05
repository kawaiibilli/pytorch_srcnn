from __future__ import print_function
import argparse
import torch
import math
from torch.autograd import Variable
from PIL import Image

from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
import numpy as np

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--input_image', type=str, required=True, help='input image to use')
parser.add_argument('--model', type=str, required=True, help='model file to use')
parser.add_argument('--output_filename', type=str, help='where to save the output image')
parser.add_argument('--scale_factor', type=float, help='factor by which super resolution needed')

parser.add_argument('--cuda', action='store_true', help='use cuda')
opt = parser.parse_args()

print(opt)

img = Image.open(opt.input_image).convert('YCbCr')
#y, cb, cr = img.split()

#tt2 = transforms.Scale((math.floor(img.size[1]*opt.scale_factor),math.floor(img.size[0]*opt.scale_factor)),interpolation=Image.CUBIC)


#img = tt2(img)

img = img.resize((int(img.size[0]*opt.scale_factor),int(img.size[1]*opt.scale_factor)),Image.BICUBIC)

model = torch.load(opt.model)
#input = Variable(ToTensor()(y)).view(1, -1, y.size[1], y.size[0])
input = Variable(ToTensor()(img)).view(1, -1, img.size[1], img.size[0])

if opt.cuda:
    model = model.cuda()
    input = input.cuda()

out = model(input)
out = out.cpu()

print ("type = ",type(out))
tt = transforms.ToPILImage()

img_out = tt(out.data[0])

img_out = img_out.convert('RGB')

img_out.save(opt.output_filename)

exit()





out_img_y = out.data[0].numpy()
out_img_y *= 255.0
out_img_y = out_img_y.clip(0, 255)
out_img_y = Image.fromarray(np.uint8(out_img_y[0]), mode='L')

out_img_cb = cb.resize(out_img_y.size, Image.BICUBIC)
out_img_cr = cr.resize(out_img_y.size, Image.BICUBIC)
out_img = Image.merge('YCbCr', [out_img_y, out_img_cb, out_img_cr]).convert('RGB')

out_img.save(opt.output_filename)
print('output image saved to ', opt.output_filename)

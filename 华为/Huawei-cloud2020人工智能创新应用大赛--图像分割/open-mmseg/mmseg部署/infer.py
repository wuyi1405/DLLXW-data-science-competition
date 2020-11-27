#from mmsegmentation.inference import inference_segmentor, init_segmentor
from mmseg.apis import inference_segmentor, init_segmentor
import mmcv
import cv2
import sys
sys.path.append('.')
from PIL import Image
import numpy as np
def main():
    # # 定义 config 文件路径
    # config_file = 'mmsegmentation/configs/deeplabv3plus/deeplabv3plus_r50-d8_512x1024_40k_cityscapes.py'
    # checkpoint_file = '../mmsegmentation/work_dirs/deeplabv3plus_r50-d8_512x1024_40k_cityscapes/latest.pth'
    #
    # # build the model from a config file and a checkpoint file
    # model = init_segmentor(config_file, checkpoint_file)
    #
    # # test a single image and show the results
    # img_path = 'demo.png'  # or img = mmcv.imread(img), which will only load it once
    # img=cv2.imread(img_path)
    # print('img_cv.shape:',img.shape)
    # # imgPil=Image.open(img_path)
    # # imgPil=np.array(imgPil)
    # # print('imgPil.shape:',imgPil.shape)
    # # img_pil2cv = cv2.cvtColor(imgPil, cv2.COLOR_RGB2BGR)
    # # print('img_pil2cv.shape:',img_pil2cv.shape)
    # result = inference_segmentor(model, img_path)
    # print(result[0])
    # #result=result[0]
    # #result=result.astype(np.int8)
    # #print(type(result[0][0]))
    # #print(result)
    # # visualize the results in a new window
    # #model.show_result(img, result, show=True)
    # # or save the visualization results to image files
    # #model.show_result(img, result, out_file='resultimg_pil2cv.jpg')
    config_file = 'config_huawei/deeplabv3plus512x512.py'
    checkpoint_file = '/home/admins/qyl/huawei_compete/mmsegmentation/work_dirs/deeplabv3plus_r50-d8_512x1024_40k_cityscapes/model_best.pth'

    # build the model from a config file and a checkpoint file
    model = init_segmentor(config_file, checkpoint_file, device='cuda')

    # test a single image and show the results
    img_path = './demo_test/182_7_26.png'  # or img = mmcv.imread(img), which will only load it once
    #img=cv2.imread(img_path)
    #print('img_cv.shape:',img.shape)
    #img = mmcv.imread(img_path)
    #(img.shape)
    imgPil = Image.open(img_path)
    imgPil = np.array(imgPil)
    print('imgPil.shape:', imgPil.shape)
    img_pil2cv = cv2.cvtColor(imgPil, cv2.COLOR_RGB2BGR)
    print('img_pil2cv.shape:', img_pil2cv.shape)
    result = inference_segmentor(model, imgPil)
    # result = result[0]
    # result = result.astype(np.int8)
    # print(type(result[0][0]))
    # print(result)
    model.show_result(imgPil, result, out_file='result1126_rgb.jpg')


if __name__ == '__main__':
    main()
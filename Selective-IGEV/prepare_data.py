import cv2
import os
import sys
import numpy as np
from PIL import Image


def run(target_size,src_dir,dst_dir,pic_ext):
    ## ---------------------------------------#
    #   一次只操作一张图片
    ## ---------------------------------------#
    for src_name in sorted(os.listdir(src_dir)):
        ## -----------------------------#
        #   把图片路径拼出来
        ## -----------------------------#
        src_file = os.path.join(src_dir, src_name)

        # img = cv2.imread(src_file)
        # original_h, original_w = img.shape[:2]
        img = Image.open(src_file)
        original_w, original_h = img.size

        img.resize(target_size,target_size)
        # # 2. 根据原始宽高比调整图像大小
        # aspect_ratio = original_w / original_h
        #
        # # 计算新的宽度和高度，保持最大边为 net_w 或 net_h
        #
        # # 裁剪为正方形
        # if aspect_ratio > 1:  # 宽图像
        #     cut_width = (original_w - original_h) // 2
        #     img  = img.crop((cut_width, 0, original_w -cut_width, original_h))
        # else:  # 高图像
        #     cut_height= (original_h - original_w) // 2
        #     img  = img.crop((0, cut_height, original_w, original_h - cut_height))

        ## -----------------------------#
        ##  opencv实现预处理
        ## -----------------------------#
        new_w=int(target_size)
        new_h=int(target_size)
        img = cv2.resize(np.array(img), (new_w, new_h), interpolation=cv2.INTER_CUBIC).astype(np.float32)
        print(f"Resized to {new_w}x{new_h} before entering the encoder.")
        ## -----------------------------------------------------#
        #   PC端网络训练时，数据需要归一化，为何在这儿不做？
        #   答：模型转换时，需要的图像输入分为是0~255，归一化操作会集成
        #       在yaml文件中mean和scale中，故不要归一化。
        ## -----------------------------------------------------#
        # img /= 255.0

        # ---------------------------------------#
        #   常规操作是：先转成RGB，再减均值，除方差
        #   到底要不要转成rgb，主要看，模型训练时用的是啥，
        #   毕竟在后面yaml配置中，这两个参数都行
        # ---------------------------------------#
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ## -----------------------------------------------------------------------------------------#
        #   PC端网络训练时，数据需要减均值，除方差，为何在这儿不做？
        #   答：为了和yaml中data_mean_and_scale下的mean_value与scale_value参数配合
        #       在yaml文件中设置即可
        ## -----------------------------------------------------------------------------------------#
        # img -= [0.485, 0.456, 0.406]
        # img /= [0.229, 0.224, 0.225]

        ## ------------------------------------------------------------------#
        #   从HWC，变为CHW。用的是Pytorch框架，其输入是NCHW，故需要这一步。
        ## ------------------------------------------------------------------#
        img = img.transpose(2, 0, 1)
        # ---------------------------------------#
        #   添加batch维度，实际运用时，有没有这一行都行
        #   至此，图像预处理完毕
        # ---------------------------------------#
        # img = np.expand_dims(img, 0)
        # img = img[np.newaxis,:,:,:]

        # -----------------------------------------------------#
        #   os.path.basename：返回最后的 文件名
        #   例如：os.path.basename("./src/1.jpg")，返回：1.jpg
        # -----------------------------------------------------#
        filename = os.path.basename(src_file)
        # print(src_file)

        # -----------------------------------------------------#
        #   os.path.splitext: 把图片名和图片扩展名分开，
        #   例如：1.jpg，short_name=1, ext=.jpg
        # -----------------------------------------------------#
        short_name, ext = os.path.splitext(filename)

        # ---------------------------------------#
        #   新的图片名
        # ---------------------------------------#
        pic_name = os.path.join(dst_dir, short_name + pic_ext)
        dtype = np.float32
        img.astype(dtype).tofile(pic_name)
        print("write:%s" % pic_name)


def main():
    # 尺寸参数
    # print(str(sys.argv))

    target_size = sys.argv[1]

    ## ------------------------------------------------------------#
    #   src_dir：原始jpg图片
    #   dst_dir：处理后的图片存放的路径
    #   pic_ext：处理后的图片后缀名(影响不大，只是为了说明它的通道顺序)
    ## ------------------------------------------------------------#
    src_dir = './origin_image'
    dst_dir = './seg_converted_{}_rgb_f32'.format(target_size)   # yaml文件中cal_data_dir参数配置成这个路径即可
    pic_ext = '.rgb'

    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)

    run(target_size,src_dir,dst_dir,pic_ext)

if __name__ == "__main__":
    main()

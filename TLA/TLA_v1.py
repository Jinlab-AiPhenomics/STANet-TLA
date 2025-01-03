import numpy as np
from PIL import ImageFile, Image
import math
import random

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

def generate_random_rgb():
    # print("generate_random_rgb")

    r = random.randint(0, 255)  # 随机生成红色值
    g = random.randint(0, 255)  # 随机生成绿色值
    b = random.randint(0, 255)  # 随机生成蓝色值
    return (r, g, b)



# 上下两行取线   理论40行++
def TLA(img):

    size = img.size  # 列(宽) 行(高)
    pxo = img.load()

    img_line = Image.new('RGB', (size[0], size[1]))  # 创建一个大图模板
    img_line.paste(img,(0,0))
    px = img_line.load()

    moulie=int(size[0]/2)

    # 1.构建水平牵引线
    # 1.1 统计搜索区间内每行冠层数量
    search_range = 100 # 参数：搜索区间100pixels
    search_range_canopy_pixel=[]

    for hi in range(size[1]):
        canopy_pixel = 0
        for hj in range(-100, 100):
            if px[moulie + hj, hi] == (128, 0, 0):  # (128, 0, 0, 255)
                canopy_pixel+=1
        search_range_canopy_pixel.append(canopy_pixel)

    # 1.2 寻找所有冠层行
    canopy_row_line = []
    row_range = 10 # 参数：每个冠层行间隔大于10pixels

    line_now=0
    for hii in range(1, len(search_range_canopy_pixel)):
        if search_range_canopy_pixel[hii - 1] == 0 and search_range_canopy_pixel[hii] > 0:
            if hii-1-line_now>row_range:
                canopy_row_line.append(hii-1)
                line_now=hii-1
        if search_range_canopy_pixel[hii - 1] > 0 and search_range_canopy_pixel[hii] == 0:
            if hii - line_now > row_range:
                canopy_row_line.append(hii)
                line_now = hii
    # 可视化 demo1_canopy_row_line 蓝线参考
    # for jj in range(len(canopy_row_line)):
    #     for right in range(size[0]):
    #         px[right, canopy_row_line[jj]] = (0,0, 255, 255) # 蓝线参考
    #
    # img.save(r'E:\Project\STANet-TLA\TLA\data\demo1_canopy_row_line.png')
    # print(len(canopy_row_line))

    # 1.3 输入小区行数先验知识 保留小区行
    plot_row_numb=2 # 参数：小区行数
    plot_row_line = [canopy_row_line[i] for i in range(len(canopy_row_line)) if (i % plot_row_numb*2 == 0) or (i % plot_row_numb*2 == plot_row_numb*2-1) or (i % plot_row_numb*2 == plot_row_numb*2)]

    # 可视化 demo1_plot_row_line_temp 蓝线参考
    # for jj in range(len(plot_row_line)):
    #     for right in range(size[0]):
    #         px[right, plot_row_line[jj]] = (0, 0, 255) # 蓝线参考
    # img.save(r'E:\Project\STANet-TLA\TLA\data\demo1_plot_row_line_temp.png')
    # print(len(plot_row_line))

    # 1.4 构造水平牵引线
    vjjj=0
    plot_row_line_new=[]
    while vjjj <len(plot_row_line):
        if vjjj==0 or vjjj==len(plot_row_line)-1:
            if vjjj==0:
                plot_row_line_new.append(plot_row_line[vjjj]-20)
                for right in range(size[0]):
                    px[right, plot_row_line[vjjj] - 20] = (0, 255, 0)  # 绿线
            elif vjjj==len(plot_row_line)-1:
                plot_row_line_new.append(plot_row_line[vjjj]+50)
                for right in range(size[0]):
                    px[right, plot_row_line[vjjj]+50] = (0, 255, 0) # 绿线
            vjjj+=1
        else:
            # for kk in range (plot_row_line[jj],plot_row_line[jj+1]):
            #     for iii in range(30):
            #         px[moulie + iii, kk] = (0, 255, 0)  # 绿块参考
            for right in range(size[0]):
                px[right, (plot_row_line[vjjj+1]-plot_row_line[vjjj])/2+plot_row_line[vjjj]] = (0, 255,0,255) # 绿线
            plot_row_line_new.append(int((plot_row_line[vjjj+1]-plot_row_line[vjjj])/2+plot_row_line[vjjj]))
            vjjj+=2


    # img_line.save(r'E:\Project\STANet-TLA\TLA\data\demo1_plot_row_line.png')
    print(len(plot_row_line_new)-1)



    # 2. 构建垂直牵引线
    # 2.1 基于水平牵引线， 统计1-2间冠层数量，
    search_canopy_col_canopy_pixel=[]
    for vi in range (len(plot_row_line_new)-1):
        canopy_col_canopy_pixel_temp=[]
        for vj in range(size[0]):
            canopy_pixel = 0
            for vk in range(plot_row_line_new[vi], plot_row_line_new[vi+1]):
                if px[vj, vk] == (128, 0, 0):  # (128, 0, 0)
                    canopy_pixel += 1
            canopy_col_canopy_pixel_temp.append(canopy_pixel)
        search_canopy_col_canopy_pixel.append(canopy_col_canopy_pixel_temp)


    # 2.2 基于小区行长先验，保留小区列边界

    canopy_col_line = []
    col_range = 200  # 参数：每个冠层行长大于pixels
    col_plot_range=10# 参数：每个校区列间距大于pixels


    # BUG: 列间距合并后如何？
    for vii in range(len(search_canopy_col_canopy_pixel)):
        vjj = 0
        canopy_col_line_temp=[]
        start=0
        while vjj < len(search_canopy_col_canopy_pixel[vii]):
            if search_canopy_col_canopy_pixel[vii][vjj - 1] == 0 and search_canopy_col_canopy_pixel[vii][vjj] > 0 and start==0: # 先黑后红，小区间隔+行长
                canopy_col_line_temp.append(vjj - 1)
                vjj+=col_range
                start=1
            elif search_canopy_col_canopy_pixel[vii][vjj - 1] > 0 and search_canopy_col_canopy_pixel[vii][vjj] == 0 and start==1:# 先红后黑，小区行长+间隔
                canopy_col_line_temp.append(vjj)
                vjj+=col_plot_range
                start=0
                # vjj+=1
            else:
                vjj += 1

        canopy_col_line.append(canopy_col_line_temp)


    # # 可视化 search_canopy_col_canopy_pixel 蓝线参考
    # for vvi in range(len(canopy_col_line)):
    #     for vvj in range(len(canopy_col_line[vvi])):
    #         for down in range(plot_row_line_new[vvi], plot_row_line_new[vvi + 1]):
    #             px[canopy_col_line[vvi][vvj], down] = (0, 0, 255) # 蓝线参考
    # img.save(r'E:\Project\STANet-TLA\TLA\data\demo1_plot_row_line_temp.png')

    # 2.3 取中，构建垂直牵引线
    plot_col_line_new=[]
    for viii in range(len(canopy_col_line)):
        vjjj=0
        plot_col_line_new_temp = []
        while vjjj <len(canopy_col_line[viii]):
            if vjjj==0 or vjjj==len(canopy_col_line[viii])-1:
                for down in range(plot_row_line_new[viii], plot_row_line_new[viii + 1]):
                    px[canopy_col_line[viii][vjjj], down] = (255, 255, 0) # 黄线
                plot_col_line_new_temp.append(int(canopy_col_line[viii][vjjj]))
                vjjj+=1
            else:
                for down in range(plot_row_line_new[viii], plot_row_line_new[viii + 1]):
                    px[(canopy_col_line[viii][vjjj+1]-canopy_col_line[viii][vjjj])/2+canopy_col_line[viii][vjjj], down] = (255, 255,0) # 黄线
                plot_col_line_new_temp.append(int((canopy_col_line[viii][vjjj+1]-canopy_col_line[viii][vjjj])/2+canopy_col_line[viii][vjjj]))
                vjjj+=2
        plot_col_line_new.append(plot_col_line_new_temp)

    print(len(plot_col_line_new))


    # # 3 微调
    # # 3.1 微调水平线
    # for vk in range (1,len(plot_row_line_new)-1):
    #     for vl in range(size[0]):
    #         if pxo[vl,vk]==(128, 0, 0): #如果压盖
    #             for vm in range(plot_row_line_new[vk-1], plot_row_line_new[vk+1]):

    # 3.2 微调垂直线


    #  # 4 小区分割
    plot_numb=0
    view_rgb=generate_random_rgb()
    img_plots = Image.new('I;16', (size[0], size[1]))  # 创建一个大图模板
    pxplot=img_plots.load()
    img_plots_view = Image.new('RGB', (size[0], size[1]))  # 创建一个大图模板
    pxplot_view=img_plots_view.load()
    for psi in range (len(plot_col_line_new)-1):
        for psj in range(len(plot_col_line_new[psi])-1):
            for pr in range(plot_row_line_new[psi],plot_row_line_new[psi+1]):
                for pc in range(plot_col_line_new[psi][psj],plot_col_line_new[psi][psj+1]):

                    if px[pc,pr]==(128, 0, 0):
                        pxplot[pc,pr]=plot_numb
                        pxplot_view[pc,pr]=view_rgb
            plot_numb+=1
            view_rgb = generate_random_rgb()

    angle1 = 1.4
    img_plots = img_plots.rotate(angle1)
    img_plots.save(r'E:\Project\STANet-TLA\TLA\data\demo-2_plots.png')

    img_plots_view = img_plots_view.rotate(angle1)
    img_plots_view.save(r'E:\Project\STANet-TLA\TLA\data\demo-2_plots_view.png')

    img_line = img_line.rotate(angle1)
    img_line.save(r'E:\Project\STANet-TLA\TLA\data\demo-2_plot_row_line.png')

    print(plot_numb + 1)



    return img_line,plot_row_line_new, plot_col_line_new


if __name__ == "__main__":
    img1 = Image.open(r"E:\Project\STANet-TLA\TLA\data\demo-2.png")


    angle1 = -1.4
    img1 = img1.rotate(angle1)


    imgline, line_row, img_col = TLA(img1)



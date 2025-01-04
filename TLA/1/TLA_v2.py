import numpy as np
from PIL import ImageFile, Image
import math
import random

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


def GSDtoPixels(GSD, length):
    pixels = length / GSD
    return int(pixels)


def rotate_map(img0, angle):
    print("rotate_map")

    angle_in_radians = math.radians(angle)
    rotate_map = img0.rotate(angle_in_radians)

    return rotate_map


def generate_random_rgb():
    # print("generate_random_rgb")

    r = random.randint(0, 255)  # 随机生成红色值
    g = random.randint(0, 255)  # 随机生成绿色值
    b = random.randint(0, 255)  # 随机生成蓝色值
    return (r, g, b)


def visuall_TLs(img, size, plot_row_line=[], plot_col_line=[]):
    img_line = Image.new('RGB', (size[0], size[1]))  # 创建一个大图模板
    img_line.paste(img, (0, 0))
    px = img_line.load()

    if len(plot_col_line) > 1:
        for vvi in range(len(plot_col_line)):
            for vvj in range(len(plot_col_line[vvi])):
                for down in range(plot_row_line[vvi], plot_row_line[vvi + 1]):
                    px[plot_col_line[vvi][vvj], down] = (0, 0, 255)  # 蓝线参考
    elif len(plot_row_line) > 1:
        for jj in range(len(plot_row_line)):
            for right in range(size[0]):
                px[right, plot_row_line[jj]] = (0, 0, 255)  # 蓝线参考

    return img_line


def get_canopy_number(px, size, moulie,search_range):
    # print("get_canopy_number")

    # 1.1 统计搜索区间内每行冠层像素数量
    search_range = search_range  # 参数：搜索区间100pixels

    moulie=int(size[0]*moulie)

    search_range_canopy_pixel = []

    for hi in range(size[1]):
        canopy_pixel = 0
        for hj in range(-search_range, search_range):
            if px[moulie + hj, hi] == (128, 0, 0):  # (128, 0, 0, 255)
                canopy_pixel += 1
        search_range_canopy_pixel.append(canopy_pixel)

    # 1.2 寻找所有冠层行
    canopy_row_line = []
    row_range = 10  # 参数：每个冠层行间隔大于10pixels

    line_now = 0
    for hii in range(1, len(search_range_canopy_pixel)):
        if search_range_canopy_pixel[hii - 1] == 0 and search_range_canopy_pixel[hii] > 0:
            if hii - 1 - line_now > row_range:
                canopy_row_line.append(hii - 1)
                line_now = hii - 1
        if search_range_canopy_pixel[hii - 1] > 0 and search_range_canopy_pixel[hii] == 0:
            if hii - line_now > row_range:
                canopy_row_line.append(hii)
                line_now = hii

    canopy_numb = math.ceil(len(canopy_row_line) / 2)


    return canopy_row_line, canopy_numb


def get_HTLs(img, px, size, plot_row_numb, moulie,search_range):
    print("get_HTLs")

    # 1.1 统计搜索区间内每行冠层数量
    canopy_row_line = get_canopy_number(px, size, moulie,search_range)[0]
    # visuall_TLs(img, size, canopy_row_line).save("E:\Project\STANet-TLA\TLA\data\\111.png")


                  # 1.2 输入小区行数先验知识 保留小区行
    plot_row_line = [canopy_row_line[i] for i in range(len(canopy_row_line)) if
                     (i % plot_row_numb * 2 == 0) or (i % plot_row_numb * 2 == plot_row_numb * 2 - 1) or (
                             i % plot_row_numb * 2 == plot_row_numb * 2)]



    # 1.3 构造水平牵引线
    vjjj = 0
    plot_row_line_new = []
    while vjjj < len(plot_row_line):
        if vjjj == 0 or vjjj == len(plot_row_line) - 1:
            if vjjj == 0:
                plot_row_line_new.append(plot_row_line[vjjj] - 50)
                for right in range(size[0]):
                    px[right, plot_row_line[vjjj] - 20] = (0, 255, 0)  # 绿线
            elif vjjj == len(plot_row_line) - 1:
                plot_row_line_new.append(plot_row_line[vjjj] + 100)
                for right in range(size[0]):
                    px[right, plot_row_line[vjjj] + 100] = (0, 255, 0)  # 绿线
            vjjj += 1
        else:
            for right in range(size[0]):
                px[right, (plot_row_line[vjjj + 1] - plot_row_line[vjjj]) / 2 + plot_row_line[vjjj]] = (
                    0, 255, 0, 255)  # 绿线
            plot_row_line_new.append(int((plot_row_line[vjjj + 1] - plot_row_line[vjjj]) / 2 + plot_row_line[vjjj]))
            vjjj += 2

    # img_line.save(r'E:\Project\STANet-TLA\TLA\data\demo1_plot_row_line.png')
    print("每列小区数：", len(plot_row_line_new) - 1)

    return plot_row_line_new


def get_VTLs(img, px, size, plot_row_line_new, plot_row_length, plot_row_space):
    print("get_VTLs")

    # 2.1 基于水平牵引线， 统计1-2间冠层数量，
    search_canopy_col_canopy_pixel = []
    for vi in range(len(plot_row_line_new) - 1):
        canopy_col_canopy_pixel_temp = []
        for vj in range(size[0]):
            canopy_pixel = 0
            for vk in range(plot_row_line_new[vi], plot_row_line_new[vi + 1]):
                if px[vj, vk] == (128, 0, 0):  # (128, 0, 0)
                    canopy_pixel += 1
            canopy_col_canopy_pixel_temp.append(canopy_pixel)
        search_canopy_col_canopy_pixel.append(canopy_col_canopy_pixel_temp)

    # 2.2 基于小区行长先验，保留小区列边界

    canopy_col_line = []
    col_range = plot_row_length  # 参数：每个冠层行长大于pixels
    col_plot_range = plot_row_space  # 参数：每个校区列间距大于pixels

    # BUG: 列间距粘连后如何？
    for vii in range(len(search_canopy_col_canopy_pixel)):
        vjj = 0
        canopy_col_line_temp = []
        start = 0
        while vjj < len(search_canopy_col_canopy_pixel[vii]):
            if search_canopy_col_canopy_pixel[vii][vjj - 1] == 0 and search_canopy_col_canopy_pixel[vii][vjj] > 0 \
                    and start == 0:  # 先黑后红，小区间隔+行长
                canopy_col_line_temp.append(vjj - 1)
                vjj += col_range
                start = 1
            elif search_canopy_col_canopy_pixel[vii][vjj - 1] > 0 and search_canopy_col_canopy_pixel[vii][
                vjj] == 0 and start == 1:  # 先红后黑，小区行长+间隔
                canopy_col_line_temp.append(vjj)
                vjj += col_plot_range
                start = 0
            else:
                vjj += 1

        canopy_col_line.append(canopy_col_line_temp)

    # # 可视化 search_canopy_col_canopy_pixel 蓝线参考
    # visuall_TLs(img,size,plot_row_line_new,canopy_col_line).save(out_path+"plot_row_line_temp.png")

    # 2.3 取中，构建垂直牵引线
    plot_col_line_new = []
    for viii in range(len(canopy_col_line)):
        vjjj = 0
        plot_col_line_new_temp = []
        while vjjj < len(canopy_col_line[viii]):
            if vjjj == 0 or vjjj == len(canopy_col_line[viii]) - 1:
                for down in range(plot_row_line_new[viii], plot_row_line_new[viii + 1]):
                    px[canopy_col_line[viii][vjjj], down] = (255, 255, 0)  # 黄线
                plot_col_line_new_temp.append(int(canopy_col_line[viii][vjjj]))
                vjjj += 1
            else:
                for down in range(plot_row_line_new[viii], plot_row_line_new[viii + 1]):
                    px[(canopy_col_line[viii][vjjj + 1] - canopy_col_line[viii][vjjj]) / 2 + canopy_col_line[viii][
                        vjjj], down] = (255, 255, 0)  # 黄线
                plot_col_line_new_temp.append(int(
                    (canopy_col_line[viii][vjjj + 1] - canopy_col_line[viii][vjjj]) / 2 + canopy_col_line[viii][vjjj]))
                vjjj += 2
        plot_col_line_new.append(plot_col_line_new_temp)


    print("每行小区数：", len(plot_col_line_new[0])-1)

    return plot_col_line_new


def get_canopy_number_v2(px, vl, vk, plot_row_line_up, plot_row_line_down):
    # print("get_canopy_number_v2")

    r_up = 0
    r_down = 0

    canopy_up = []
    canopy_down = []
    for ii in range(plot_row_line_up, plot_row_line_down):
        if ii - vk < 0:
            canopy_up.append(px[vl, ii])
        else:
            canopy_down.append(px[vl, ii])

    for jjj in range(len(canopy_up) - 1):
        if canopy_up[jjj] == 0 and canopy_up[jjj + 1] == 1:
            r_up += 1
        if canopy_up[jjj] == 1 and canopy_up[jjj + 1] == 0:
            r_up += 1

    for jjj in range(len(canopy_down) - 1):
        if canopy_down[jjj] == 0 and canopy_down[jjj + 1] == 1:
            r_down += 1
        if canopy_down[jjj] == 1 and canopy_down[jjj + 1] == 0:
            r_down += 1

    return math.ceil(r_up / 2), math.ceil(r_down / 2)

def refine_HTLs(px,px_ori, size, plot_row_line):
    print("refine_HTLs")

    plot_row_line_new = []

    # 3.1.1 查找是否压盖需要移动
    for vk in range(0, len(plot_row_line)):  # plot_row_line_new 1行n列，每列一个值，为绿线纵坐标，更新所有纵坐标
        plot_row_line_new_temp = []
        for vl in range(size[0]):
            if px_ori[vl, plot_row_line[vk]] == (128, 0, 0) and vk>0 and vk<len(plot_row_line)-1:  # 如果压盖
                px[vl, plot_row_line[vk]] = (128, 0, 0)
                # 3.1.2统计两侧冠层数量和需要移动的距离
                # 统计方向
                r_up, r_down = get_canopy_number_v2(px_ori, vl, plot_row_line[vk], plot_row_line[vk - 1], plot_row_line[vk + 1])
                # 统计距离
                d_up = 0
                d_down = 0
                while px_ori[vl, plot_row_line[vk] - d_up] == (128, 0, 0):
                    d_up += 1
                while px_ori[vl, plot_row_line[vk] + d_down] == (128, 0, 0):
                    d_down += 1

                # print(d_up,d_down)
                if r_up > r_down:
                    plot_row_line_new_temp.append(plot_row_line[vk] - d_up)
                    px[vl, plot_row_line[vk] - d_up] =  (0, 255, 0)

                else:
                    plot_row_line_new_temp.append(plot_row_line[vk] + d_down)
                    px[vl, plot_row_line[vk] + d_down] = (0, 255, 0)

            else:
                if px_ori[vl, plot_row_line[vk]] == (128, 0, 0) and vk==0:
                    px[vl, plot_row_line[vk]] = (128, 0, 0)

                    d_up = 0
                    while px_ori[vl, plot_row_line[vk] - d_up] == (128, 0, 0):
                        d_up += 1
                    plot_row_line_new_temp.append(plot_row_line[vk] - d_up)
                    px[vl, plot_row_line[vk] - d_up] =  (0, 255, 0)

                if px_ori[vl, plot_row_line[vk]] == (128, 0, 0) and vk == len(plot_row_line)-1:
                    px[vl, plot_row_line[vk]] = (128, 0, 0)

                    d_down = 0
                    while px_ori[vl, plot_row_line[vk] + d_down] == (128, 0, 0):
                        d_down += 1
                    plot_row_line_new_temp.append(plot_row_line[vk] + d_down)
                    px[vl, plot_row_line[vk] + d_down] = (0, 255, 0)


                plot_row_line_new_temp.append(plot_row_line[vk])

        plot_row_line_new.append(plot_row_line_new_temp)

    return plot_row_line_new


def refine_VTLs(px,px_ori, size, plot_col_line, plot_row_line):
    print("refine_VTLs")

    plot_col_line_new = []

    # 3.2.1 查找是否压盖需要移动

    for psi in range(len(plot_col_line)):  # plot_col_line m行n列，m行小区，每行n个线，为绿线横坐标，更新所有m行内的横坐标
        plot_col_line_new_temp0 = []
        for psj in range(len(plot_col_line[psi])):
            plot_col_line_new_temp1 = []

            r_left = 0
            r_right = 0

            r_first=0
            for pr in range(plot_row_line[psi], plot_row_line[psi + 1]):  # 第m行纵坐标的区间
                if px_ori[plot_col_line[psi][psj], pr] == (128, 0, 0):  # 如果压盖
                    px[plot_col_line[psi][psj], pr] = (128, 0, 0)
                    # 统计方向和距离
                    # 统计距离
                    d_left = 0
                    d_right = 0

                    if r_first==0:
                        r_first=1

                        for ss in range(0,100):
                            first_left=0
                            first_right=0

                            while px_ori[plot_col_line[psi][psj] - first_left+ss, pr] == (128, 0, 0):

                                first_left+= 1
                                if r_left<first_left:
                                    r_left=first_left
                                if ss==0:
                                    d_left += 1
                            while px_ori[plot_col_line[psi][psj] + first_right+ss, pr] == (128, 0, 0):

                                first_right += 1
                                if r_right < first_right:
                                    r_right = first_right
                                if ss==0:
                                    d_right += 1


                    if r_left > r_right: # 方向
                        plot_col_line_new_temp1.append(plot_col_line[psi][psj] + d_right)
                        px[plot_col_line[psi][psj] + d_right, pr] = (255, 255, 0)

                    else:
                        plot_col_line_new_temp1.append(plot_col_line[psi][psj] - d_left)
                        px[plot_col_line[psi][psj] - d_left, pr] = (255, 255, 0)


                else:
                    plot_col_line_new_temp1.append(plot_col_line[psi][psj])
            plot_col_line_new_temp0.append(plot_col_line_new_temp1)
        plot_col_line_new.append(plot_col_line_new_temp0)

    return plot_col_line_new


def seg_plots(px,px_ori, size, plot_row_line_new, plot_col_line_new):
    print("seg_plots")

    plot_numb = 0
    plot_rgb = generate_random_rgb()
    img_plots = Image.new('I;16', (size[0], size[1]))  # 创建一个大图模板
    pxplot = img_plots.load()

    img_plots_view = Image.new('RGB', (size[0], size[1]))  # 创建一个大图模板
    pxplot_view = img_plots_view.load()

    for psi in range(len(plot_col_line_new) - 1):
        for psj in range(len(plot_col_line_new[psi]) - 1):
            for pr in range(plot_row_line_new[psi], plot_row_line_new[psi + 1]):
                for pc in range(plot_col_line_new[psi][psj], plot_col_line_new[psi][psj + 1]):

                    if px_ori[pc, pr] == (128, 0, 0):
                        pxplot[pc, pr] = plot_numb
                        pxplot_view[pc, pr] = plot_rgb

            plot_numb += 1
            plot_rgb = generate_random_rgb()
            print(plot_numb)
    print(plot_numb + 1)

    return img_plots, img_plots_view


def seg_plots_v2(px,px_ori, size, plot_row_line_new, plot_col_line_new, plot_row_line):
    print("seg_plots_v2")
    # print(plot_row_line_new[0])
    # print(plot_row_line)


    plot_numb = 0
    plot_rgb = generate_random_rgb()
    img_plots = Image.new('I;16', (size[0], size[1]))  # 创建一个大图模板
    pxplot = img_plots.load()

    img_plots_view = Image.new('RGB', (size[0], size[1]))  # 创建一个大图模板
    pxplot_view = img_plots_view.load()

    # 两次补全
    for psi in range(len(plot_col_line_new)):
        for psj in range(len(plot_col_line_new[psi]) - 1):
            for pr in range(plot_row_line[psi], plot_row_line[psi + 1]):

                for pskl in range(plot_col_line_new[psi][psj][pr-plot_row_line[psi]], plot_col_line_new[psi][psj + 1][pr-plot_row_line[psi]]):


                    if px_ori[pskl, pr] == (128, 0, 0):
                        pxplot[pskl, pr] = plot_numb
                        pxplot_view[pskl, pr] = plot_rgb


                    #上一编号和颜色
                    pre_number=0
                    pre_rgb=(0,0,0)

                    if psi>0:

                        centerpskl=int((plot_col_line_new[psi][psj][pr-plot_row_line[psi]]+plot_col_line_new[psi][psj + 1][pr-plot_row_line[psi]])/2)
                        centerpr=int((plot_row_line[psi-1]+plot_row_line[psi])/2)
                        aa=0
                        for sss in range(-50,50):
                            for qqq in range(-50,50):
                                if px_ori[centerpskl+sss, centerpr+qqq] == (128, 0, 0):
                                    pre_number=pxplot[centerpskl+sss, centerpr+qqq]
                                    pre_rgb=pxplot_view[centerpskl+sss, centerpr+qqq]
                                    aa=1
                                    break
                            if aa==1:
                                break

                        # print(psi,psj,centerpskl,centerpr,pre_number,pre_rgb,pskl, pr)




                    # 小区结束 重新找纵向优化
                    if pskl==plot_col_line_new[psi][psj + 1][pr-plot_row_line[psi]]-1 and pr== plot_row_line[psi + 1]-1:
                        # 第一条线
                        w1=plot_col_line_new[psi][psj][pr-plot_row_line[psi]]
                        w2=plot_col_line_new[psi][psj + 1][pr-plot_row_line[psi]]-1

                        h1=plot_row_line[psi]
                        h2=plot_row_line[psi+1]

                        for psklll in range(w1,w2):
                            # print(plot_row_line_new[psi][psklll],plot_row_line[psi])
                            if plot_row_line_new[psi][psklll] < plot_row_line[psi]: #如果小于，上凸
                                for prup in range(plot_row_line_new[psi][psklll], h1):
                                    if px_ori[psklll, prup] == (128, 0, 0):
                                        pxplot[psklll, prup] = plot_numb
                                        pxplot_view[psklll, prup] = plot_rgb
                            if plot_row_line_new[psi][psklll] > plot_row_line[psi]:
                                for prup in range(h1, plot_row_line_new[psi][psklll]):
                                    if px_ori[psklll, prup] == (128, 0, 0):
                                        pxplot[psklll, prup] = pre_number
                                        pxplot_view[psklll, prup] = pre_rgb
                            # 最后一条线
                            if plot_row_line_new[psi + 1][psklll] > plot_row_line[psi + 1]:  # 如果大于，下凸
                                for prdown in range(h2, plot_row_line_new[psi + 1][psklll]+1):

                                    if px_ori[psklll, prdown] == (128, 0, 0):
                                        pxplot[psklll, prdown] = plot_numb
                                        pxplot_view[psklll, prdown] = plot_rgb

                            if plot_row_line_new[psi + 1][psklll] < plot_row_line[psi + 1]:
                                for prdown in range(plot_row_line_new[psi + 1][psklll], h2):
                                    if px_ori[psklll, prdown] == (128, 0, 0):
                                        pxplot[psklll, prdown] = 0
                                        pxplot_view[psklll, prdown] = (0, 0, 0)



            plot_numb += 1
            # print(plot_numb)
            plot_rgb = generate_random_rgb()

    print("小区个数：",plot_numb + 1)

    return img_plots, img_plots_view


# 上下两行取线   理论40行++
def TLA(map_file, out_path, rotate_angle, moulie, search_range,plot_row_length, plot_row_number, plot_row_space):
    ori_img = Image.open(map_file)
    img = rotate_map(ori_img, rotate_angle)
    px_ori = img.load()

    size = img.size  # 列(宽) 行(高)

    img_line = Image.new('RGB', (size[0], size[1]))  # 创建一个大图模板
    img_line.paste(img, (0, 0))
    px = img_line.load()

    # 1.构建水平牵引线
    plot_row_line = get_HTLs(img, px, size, plot_row_number, moulie,search_range)

    # 2. 构建垂直牵引线
    plot_col_line = get_VTLs(img, px, size, plot_row_line, plot_row_length, plot_row_space)

    # #  # 3 小区分割
    # img_plots,img_plots_view = seg_plots(px,px_ori, size, plot_row_line, plot_col_line)
    # rotate_map(img_line, -rotate_angle).save(out_path + 'demo2_plots_liinenno.png')

    # # 3 微调
    # # 3.1 微调水平线
    plot_row_line_newn = refine_HTLs(px,px_ori, size, plot_row_line)

    # 3.2 微调垂直线
    plot_col_line_newn = refine_VTLs(px,px_ori, size, plot_col_line, plot_row_line)


    #  # 4 小区分割
    img_plots, img_plots_view = seg_plots_v2(px,px_ori, size, plot_row_line_newn, plot_col_line_newn, plot_row_line)


    rotate_map(img_line, -rotate_angle).save(out_path + 'demo-3_plots_row_line.png')
    rotate_map(img_plots, -rotate_angle).save(out_path + 'demo-3_plots.png')
    rotate_map(img_plots_view, -rotate_angle).save(out_path + 'demo-3_plots_view.png')

    return 0


if __name__ == "__main__":
    semmap_file = r"E:\Project\STANet-TLA\TLA\data\demo-3.png"
    out_path = r"E:\Project\STANet-TLA\TLA\data\\"

    # # 2hang demo
    # GSD = 1
    # rotate_angle = -80
    # plot_row_length = 200
    # plot_row_number = 2
    # plot_row_space = 10
    # search_range=100
    # moulie = 0.55238 #搜索位置


    # 3hang demo
    GSD = 1
    rotate_angle = 0
    plot_row_length = 50
    plot_row_number = 3
    plot_row_space = 0
    search_range=10
    moulie = 0.449038


    runs = TLA(semmap_file, out_path, rotate_angle, moulie,search_range,GSDtoPixels(GSD, plot_row_length), plot_row_number,
               GSDtoPixels(GSD, plot_row_space))



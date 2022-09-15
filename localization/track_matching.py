

import cv2
import numpy as np
import math
import argparse
import cv2
import time
import glob
import os, subprocess

from collections import Counter

rn_nm = 0

def gplot(d_src,d_dst):
    stc = d_src+"\\*.*"
    dstc = d_dst+"\\*.*"
    b_thres = int(opt.threshold)
    src = glob.glob(stc)
    dst = glob.glob(dstc)
    dc_copy = dst.copy()
    rs_ts = []
    rec_t = []
    for i in src:
        rs_idx = 0
        rs_tmp = []
        rtx_index = 0
        #找到最匹配位置
        for j in range(len(dst)):
            sl_num = (os.path.basename(dst[j])).split('.')
            if not sl_num in rec_t:
                rtx_index = sift(i,dst[j])
                if rtx_index>rs_idx:
                    rs_idx = rtx_index
                    rs_tmp=(dst[j])
        if int(rs_idx)> b_thres:
            rs_tmp = rname_file(rs_tmp,i)
            #dst[j] 对比目的组图片 ,rs_tmp  ,rs_idx 当前匹配点数量
            tr = (rs_tmp)
            rs_ts.append(tr)
            rtx_tmp = (os.path.basename(i)).split('.')
            rec_t.append(rtx_tmp[0])
    # print(rec_t)
    find_new(dc_copy,rs_ts)

def find_new(dc_copy,rs_ts):
    global rn_nm
    s1 = set(dc_copy)
    s2 = set(rs_ts)
    r_sa = s1 & s2
    r_dif = s1 ^ r_sa
    # print(r_dif)
    # rn_nm = 0
    for dc_pth in dc_copy:
        l_tmp = os.path.basename(dc_pth)
        rs_nm = int(l_tmp.split('.')[0])
        if int(rs_nm) > int(rn_nm):
            rn_nm = rs_nm
    # print("\n rn_nm:")
    # print(rn_nm)
    r_df_list = list(r_dif)
    ts_num = rn_nm
    for i in r_df_list:
        ts_num+=1
        sy_path = os.path.split(os.path.realpath(i))
        r_prefix = sy_path[-1].split('.')
        rs_tmp_pth = sy_path[0]+"\\"+str(ts_num)+'.'+str(r_prefix[-1])
        print("\n new objects tracking:")
        print(ts_num,rs_tmp_pth)
        if not os.path.exists(rs_tmp_pth):
            os.rename(i,rs_tmp_pth)



def rname_file(rs_tmp,org_tmp):
    org_name = os.path.basename(org_tmp)
    org_name1t = org_name.split('.')
    rs_name = os.path.basename(rs_tmp)
    rs_path = rs_tmp.replace(rs_name,'')
    rs_name1t = rs_name.split('.')
    rs_name = str(rs_name1t[0]) +'.'+str(rs_name1t[-1])
    org_name = str(org_name1t[0])+'.'+str(org_name1t[-1])
    rs_dst = rs_path + rs_name
    org_src = rs_path + org_name
    rs_tmp = rs_path+ 'temp.jpg'
    if rs_name == org_name:
        return rs_dst
    # print(org_src,rs_dst,"\n")
    if os.path.exists(rs_tmp):
        os.remove(rs_tmp)
    if os.path.exists(org_src):
        if os.path.exists(rs_dst) and org_src != rs_dst:
            os.rename(rs_dst, rs_tmp)
            os.rename(org_src, rs_dst)
            os.rename(rs_tmp, org_src)
        elif not os.path.exists(rs_dst) or org_src == rs_dst:
            os.rename(org_src, rs_dst)
    return rs_dst


# sift comparation
def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def sift(srce,dste):
    # find correspondences using simple weighted sum of squared differences
    src = cv2.imread(srce)
    dst = cv2.imread(dste)

    dh1, dw1, _ = src.shape
    dh2, dw2, _ = dst.shape
    img1 = src
    if dw1<12 or dh1<16:
        img1 = cv2.resize(src, (12, 16))

    img2 = dst
    if dw2<12 or dh2<16:
        img2 = cv2.resize(dst, (12, 16))

    # img1 = cv2.resize(src, (160, 120))
    # img2 = cv2.resize(dst, (160, 120))

    # print(f"OpenCV Version: {cv2.__version__}")

    # cv_show("img1",img1)
    # cv_show("img2",img2)

    #检测图片
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    if des1 is None or des2 is None:
        print(des1,des2)
        return 0
    if len(des1)<3 or len(des2)<3:
        print(len(des1),len(des2))
        return 1
    # 一对一的匹配 然后他通过
    bf1 = cv2.BFMatcher(crossCheck=True)  # 选择暴力匹配，默认通过比较欧氏距离来进行匹配计算分析

    matches1 = bf1.match(des1, des2)
    matches1 = sorted(matches1, key=lambda x: x.distance)  # 通过距离度量，由最接近到最不接近进行排序

    img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches1[:10], None, flags=2)  # 取前10的进行匹配显示
    # cv2.imshow("img3",img3)

    # k对最佳匹配
    bf2 = cv2.BFMatcher()
    matches2 = bf2.knnMatch(des1, des2, k=2)  # 此时变为一个点对应两个最近的点

    good = []
    fea_goods = []
    for m, n in matches2:
        if m.distance < 0.75 * n.distance:  # 设定一种过滤方法
            good.append([m])
    for i in range(len(good)):
        fea_goods.append(good[i][0].distance)
    res_l = len(fea_goods)
    return res_l
    # img4 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)
    # cv2.imshow("img4", img4)
    # cv2.waitKey(1)


def list_dir(opt):
    tre = opt.rpath
    r_dir = (os.listdir(opt.rpath))

    # print(r_dir)
    rt = []
    for i in range(len(r_dir)):
        rt.append(int(i))
    rt = sorted(rt)
    for j in range(1,len(rt)):
        d_src = tre+str(rt[j-1])
        d_dst = tre+str(rt[j])
        # print(d_src,d_dst)
        gplot(d_src,d_dst)


def sort_opt_list(r_dir):
    stc = glob.glob(r_dir+"*")
    for i in stc:
        trc = glob.glob(i+"\\*.*")
        for j in trc:
            org_name = os.path.basename(j)
            org_name1t = org_name.split('.')
            sort_file_path = opt.rpath+"sort\\"+org_name1t[0]+"\\"
            sub_folder = os.path.dirname(j)
            sub_folder_last = sub_folder.split("\\")[-1]
            sort_last_name = sort_file_path +sub_folder_last+"_"+org_name1t[0]+"."+org_name1t[-1]
            # print(j,sort_last_name)

            if not os.path.exists(sort_file_path):
                os.makedirs(sort_file_path)
            src = os.path.realpath(j)
            dst = os.path.realpath(sort_last_name)

            status = subprocess.check_output(['copy', src, dst], shell=True)
            # print("status: ", status.decode('utf-8'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s','--src', type=str, default='data/images', help='source')  # source file/folder
    # parser.add_argument('-d','--dst', type=str, default='data/images', help='dst')  # dst file/folder
    parser.add_argument('-r','--rpath', type=str, default='data/images', help='rpath')  # root folder path
    parser.add_argument("-t", "--threshold", type=int, default=10, help="Threshold for this matching")

    t0 = time.time()
    opt = parser.parse_args()
    #test
    # gplot(opt.src,opt.dst)

    #whole process
    list_dir(opt)
    sort_opt_list(opt.rpath)
    # print(sift(opt.src,opt.dst))
    print(f'Done. ({time.time() - t0:.3f}s) ' +' \n')




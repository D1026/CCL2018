import pandas as pd
import numpy as np
import pickle
import string

classTree = {'投诉（含抱怨）': ['网络问题', '营销问题', '费用问题', '服务问题', '业务使用问题', '业务办理问题', '业务规定不满', '不知情定制问题', '信息安全问题', '电商售后问题'], \
             '办理': ['开通', '取消', '变更', '下载/设置', '转户', '打印/邮寄', '重置/修改/补发', '缴费', '移机/装机/拆机', '停复机', '补换卡', '入网', '销户/重开'], \
             '咨询（含查询）': ['产品/业务功能', '账户信息', '业务资费', '业务订购信息查询', '使用方式', '办理方式', '业务规定', '号码状态', '用户资料', '服务渠道信息', \
                         '工单处理结果', '电商货品信息', '营销活动信息', '宽带覆盖范围'], \
             '表扬及建议': ['表扬', '建议'], \
             '特殊来电': ['无声电话', '骚扰电话'], \
             '转归属地10086': '', \
             '非移动业务': '', \
             'c': '', \
             '非来电': ''}

y_dict, y1_dict = pd.read_pickle("yy_dict.pkl")
y = pd.read_pickle("predict_y.pkl")
y1 = pd.read_pickle("predict_y1.pkl")

# -------
print(y[0])
exit(888)

def softmaxToOnehont(y):
    shape = y.shape
    y_ = np.zeros(shape=shape)
    for i in range(shape[0]):
        index = np.argmax(y[i])
        y_[i][index] = 1
    return y_


y_ = softmaxToOnehont(y)
y1_ = softmaxToOnehont(y1)
print(y[0])
print(y_[0])

print(y1[0])
print(y1_[0])


def onehotToClass(y, y_dict):
    y_list = []
    shape = y.shape
    for i in range(shape[0]):
        for k in y_dict.keys():
            if (y[i] == y_dict[k]).all():
                y_list.append(k)
    return y_list


y_list = onehotToClass(y_, y_dict)
y1_list = onehotToClass(y1_, y1_dict)

# with open('y.pkl', 'wb') as f:
#     pickle.dump(y_list, f)
#
# with open('y1.pkl', 'wb') as f:
#     pickle.dump(y1_list, f)
#
#
# import pandas as pd

id = pd.read_pickle("ID.pkl")
# y = pd.read_pickle("y.pkl")
# y1 = pd.read_pickle("y1.pkl")
print('id_num: '+str(len(id)))
print('y_num: '+str(len(y_)))
print('y1_num: '+str(len(y1_)))

id_y_y1 = []
clashCount = 0
for i in range(len(id)):
    if y1_list[i] not in classTree[y_list[i]]:
        print('clash--', y_list[i], y1_list[i])
        for k in classTree.keys():
            if y1_list[i] in classTree[k]:
                y_list[i] = k
                break
        print('rectified--', y_list[i], y1_list[i])
        clashCount = clashCount + 1

    str = id[i] + "\t" + y_list[i] + "\t" + y1_list[i] + "\n"
    id_y_y1.append(str)
print('冲突数量：')
print(clashCount)

with open("./result.txt", mode="w", encoding="utf-8") as f:
    f.writelines(id_y_y1)
f.close()
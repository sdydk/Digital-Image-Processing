import cv2
import numpy as np
import matplotlib.pyplot as plt
class NMS:
    def __int__(self, center=False,scale=1.0):
        self.center = center
        self.scale = scale
    def compute_iou(self, bbox1, bbox2, eps=1e-8):
        if self.center:
            x1, y1, w1, h1 = bbox1
            xmin1, ymin1 = int(x1 - w1 / 2.0), int(y1 - h1 / 2.0)
            xmax1, ymax1 = int(x1 + w1 / 2.0), int(y1 - h1 / 2.0)
            x2, y2, w2, h2 = bbox2
            xmin2, ymin2 = int(x2 - w2 / 2.0), int(y2 - h2 / 2.0)
            xmax2, ymax2 = int(x2 + w2 / 2.0), int(y2 - h2 / 2.0)
        else:
            xmin1, ymin1, xmax1, ymax1 = bbox1
            xmin2, ymin2, xmax2, ymax2 = bbox2

        # 计算交集的对角坐标
        xx1 = np.max([xmin1, xmin2])
        yy1 = np.max([ymin1, ymin2])
        xx2 = np.min([xmax1, xmax2])
        yy2 = np.min([ymax1, ymax2])

        # 计算交集面积
        w = np.max([0.0, xx2 - xx1 + 1])
        h = np.max([0.0, yy2 - yy1 + 1])
        area_intersection = w * h

        # 计算并集面积
        area1 = (xmax1 - xmin1 + 1) * (ymax1 - ymin1 + 1)
        area2 = (xmax2 - xmin2 + 1) * (ymax2 - ymin2 + 1)
        area_union = area1 + area2 - area_intersection

        # 计算两个边框的交并比
        iou = area_intersection / (area_union + eps)

        return iou
    @classmethod
    def py_cpu_nms(cls, dets, iou_thresh = 0.5, score_thresh=0.5):
        dets = dets[np.where(dets[:, -1] >= score_thresh)[0]]
        xmin = dets[:, 0]
        ymin = dets[:, 1]
        xmax = dets[:, 2]
        ymax = dets[:, 3]
        scores = dets[:, 4]

        order = scores.argsort()[::-1] #按scores降序排序， argsort返回降序后的索引
        areas = (xmax - xmin + 1) * (ymax - ymin + 1)
        keep = []  # 保留最优的结果

        # 搜索最佳边框
        while order.size > 0:
            top1_idx = order[0]  #选取得分最高的边框
            keep.append(top1_idx) #添加到候选列表

            # 将得分最高的边框与剩余边框进行比较
            xx1 = np.maximum(xmin[top1_idx], xmin[order[1:]])
            yy1 = np.maximum(ymin[top1_idx], ymin[order[1:]])
            xx2 = np.minimum(xmax[top1_idx], xmax[order[1:]])
            yy2 = np.minimum(ymax[top1_idx], ymax[order[1:]])

            # 计算交集
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            intersection = w * h

            # 计算并集
            union = areas[top1_idx] + areas[order[1:]] - intersection

            # 计算交并比
            iou = intersection / union

            # 将重叠度大于给定阈值的边框剔除掉，仅保留余下的边框,返回相应的下标
            inds = np.where(iou <= iou_thresh)[0]

            # 从剩余边框中继续筛选
            order = order[inds + 1]

        return keep
if __name__ == '__main__':
    save_dir = 'Output'
    img = cv2.imread("Images/1689167501326.jpg")
    img_cp = np.copy(img)
    thickness = 2
    info = np.array([
        [30, 10, 200, 200, 0.95],
        [25, 15, 180, 220, 0.98],
        [35, 40, 190, 170, 0.96],
        [60, 60, 90, 90, 0.3],
        [20, 30, 40, 50, 0.1],
    ])
    colors = [[0, 0, 255], [0, 255, 0], [255, 0, 0], [255, 255, 0], [0, 255, 255]]
    plt.subplot(121)
    plt.axis('off')
    plt.title("Input image")
    for i in range(len(colors)):
        x1, y1, x2, y2, _, = info[i]
        # 在图像上画框
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), colors[i], thickness=thickness)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)

    plt.subplot(122)
    plt.axis('off')
    plt.title("After NMS")
    indx = NMS.py_cpu_nms(dets=info, iou_thresh=0.5, score_thresh=0.5)
    for i in indx:
        x1, y1, x2, y2, _ = info[i]
        cv2.rectangle(img_cp, (int(x1), int(y1)), (int(x2), int(y2)), colors[i], thickness=thickness)
    img_cp = cv2.cvtColor(img_cp, cv2.COLOR_BGR2RGB)
    plt.imshow(img_cp)

    plt.savefig(save_dir + "\\" + 'Non-Maximum Suppression (NMS).png')
    plt.show()








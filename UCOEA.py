import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from sklearnex import patch_sklearn
# patch_sklearn()
from sklearn import mixture
from sklearn import preprocessing
from sklearn.cluster import KMeans
import random

# Unsupervised Clustering Optimization-based Efficient Attention (UCOEA)
class UCOEA(nn.Module):
    def __init__(self, channel, k, n):
        super(UCOEA, self).__init__()
        self.down = nn.Conv2d(channel, channel//2, kernel_size=3, stride=2, padding=1, groups=channel//2, bias=False)  # (B,C,H,W)->(B,C/2,H/2,W/2)

        self.k = k  # 聚类数量
        self.kmeans = KMeans(n_clusters=self.k, tol=0.1)
        self.n = n  # 高斯分布数量
        self.gmm = mixture.GaussianMixture(n_components=self.n, covariance_type='diag', tol=0.1)
        self.conv = nn.Conv2d(self.k, self.k, (self.n,1), groups=self.k, bias=False)
        self.sig = nn.Sigmoid()

        self.up = nn.Conv2d(channel // 2, channel, kernel_size=1, stride=1, padding=0, groups=channel // 2, bias=False)  # (B,C/2,H/2,W/2)->(B,C,H/2,W/2)

    def forward(self, sx):
        x = self.down(sx)  # (B,C,H,W)->(B,C/2,H/2,W/2)

        b, c, h, w = x.size()  # (b,c,h,w)
        r = random.randint(0, b-1)  # 随机取出当前batch里的某一个，作为聚类的基准。每个batch必须使用完全相同的统一操作。
        xx = x[r,:,:,:].view(c,-1).detach().cpu().numpy()  # (b,c,h,w)->(c,h,w)->(c,hw)

        v2 = []  # 用来存储每类的索引
        v22 = []  # 用来存储每类均值特征的标准差
        label = self.kmeans.fit(preprocessing.minmax_scale(xx,axis=1)).labels_  # 数据进行归一化!!! (c,hw)->(c,)
        {v2.append(np.where(label == kk)) for kk in range(self.k)}  # (c,)->(xxx,)->(k,不同)
        {v22.append(np.mean(xx[v2[kk][0]],0).std()) for kk in range(self.k)}  # (c,hw)->(xxx,hw)->(hw,)->(1)->(k)
        i = np.argsort(np.array(v22))  # 每类"均值特征的标准差从小到大重新排序后"，返回原始标量索引。(k,)

        v3 = []  # 用来存储"标准差从小到大重新排序后"每类对应的索引
        v4 = []  # 用来存储"标准差从小到大重新排序后"位于每类索引的目标值
        v5 = torch.tensor([]).cuda()  # 用来存储"标准差从小到大重新排序后"每类对应的均值特征
        for kk in range(self.k):
            s = v2[i[kk]][0]  # 按"标准差从小到大重新排序后"依次拿出每类索引;(不同,)
            v3.append(s)  # (k,不同)
            ss = torch.index_select(x, 1, torch.tensor(s).cuda())  # (b,c,h,w)->(b,xxx,h,w)
            v4.append(ss)  # (k,b,不同,h,w); k个"不同"求和等于c; 某个"不同"记为xxx
            v5 = torch.cat([v5, torch.mean(ss, dim=1, keepdim=True)], dim=1)  # (b,xxx,h,w)->(b,1,h,w)->(b,k,h,w)
        wa = v5.view(b, self.k, -1).permute(0,2,1).detach().cpu().numpy()  # (b,k,h,w)->(b,k,hw)->(b,hw,k)

        v6 = torch.tensor([]).cuda()  # 用来存储"标准差从小到大重新排序后"每类均值特征EM后的均值
        v66 = torch.tensor([]).cuda()  # 用来存储"标准差从小到大重新排序后"每类均值特征EM后的方差
        for bb in range(len(wa)):
            sss = self.gmm.fit(wa[bb])  # (b,hw,k)->(hw,k)->(n,k)
            wb = np.nan_to_num(sss.means_).astype(np.float32)  # (n,k)
            wc = np.nan_to_num(sss.covariances_).astype(np.float32)  # (n,k)
            v6 = torch.cat([v6, torch.tensor(wb).cuda().permute(1,0).view(1, self.k, self.n, 1)], dim=0)  # (n,k)->(k,n)->(1,k,n,1)->(b,k,n,1)
            v66 = torch.cat([v66, torch.tensor(wc).cuda().permute(1, 0).view(1, self.k, self.n, 1)], dim=0) # (n,k)->(k,n)->(1,k,n,1)->(b,k,n,1)
        qq = torch.argsort(v66, 2)  # 方差按照大小重新排列后，返回原始索引。(b,k,n,1)

        box = torch.empty(b, self.k, self.n, 1).cuda()  # 用来存储按照方差大小重新排列的均值
        for bb in range(b):
            for kk in range(self.k):
                if qq[bb,kk,0,:] == 0:
                    box[bb,kk,:,:]=v6[bb, kk, :, :]
                if qq[bb,kk,0,:] == 1:
                    box[bb,kk,0,:]=v6[bb, kk, 1, :]
                    box[bb,kk,1,:]=v6[bb, kk, 0, :]
        y = self.sig(self.conv(box))  # 均值按照方差顺序排列;(b,k,n,1)->(b,k,1,1), 注意力

        v7 = torch.empty(b, c, h, w).cuda()  # 用来存储校准后的值,按索引依次返回
        for kk in range(self.k):
            v7[:,v3[kk],:,:] = y[:,kk,:,:].unsqueeze(1) * v4[kk]  # (b,k,1,1)->(b,1,1)->(b,1,1,1)*(b,xxx,h,w)->(b,xxx,h,w)

        v7 = F.interpolate(self.up(v7+x), size=(h * 2, w * 2), mode='bilinear', align_corners=True)+sx  # (B,C/2,H/2,W/2)->(B,C,H/2,W/2)->(B,C,H,W)

        return v7  # (b,c,h,w)





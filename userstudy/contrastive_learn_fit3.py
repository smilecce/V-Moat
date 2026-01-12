# 加 threshold的代码情况
import torch
import torch.nn as nn
from torch.nn import init
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import pandas as pd
import torchaudio
from certification.audibility import Audibility_new
from tqdm import tqdm
import math
import time
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def wav_padding_2048(wav_x):
    if wav_x.size(1)%2048 != 0:
        wav_x = wav_x[:, :-(wav_x.size(1)%2048)]
    return wav_x

def min_max_normalize(data):
    """
    最大-最小值归一化
    """
    min_val = min(data)
    max_val = max(data)
    normalized_data = [5 * (x - min_val) / (max_val - min_val) for x in data]
    return normalized_data




# # 加载数据集，并分成两个子集
csv_path1 = 'path1.csv'
csv_path2 = 'path2.csv'
csv_path3 = 'path3.csv'





# 选取每个人得分的最大值和每个人得分的最小值。
all_high_data = pd.DataFrame()
all_low_data = pd.DataFrame()

threshold = 1
# 邓江毅
ans = pd.read_csv(csv_path1)
ans['quality'] = min_max_normalize(ans['quality'])
ans = ans[(ans['audibility2']!=0)]    
for i in range(0, len(ans)):
    for j in range(0, len(ans)):  
        if ans.iloc[j, 6] - ans.iloc[i, 6] > threshold:  #quality [i, 6]
            all_low_data = pd.concat([all_low_data,pd.DataFrame(ans.iloc[i,:]).T], ignore_index = True)
            all_high_data = pd.concat([all_high_data,pd.DataFrame(ans.iloc[j,:]).T], ignore_index = True)




# 吴嘉林
ans = pd.read_csv(csv_path2) 
ans['quality'] = min_max_normalize(ans['quality'])
ans = ans[(ans['audibility2']!=0)]    
for i in range(0, len(ans)):
    for j in range(0, len(ans)):  
        if ans.iloc[j, 6] - ans.iloc[i, 6] > threshold:  #quality [i, 6]
            all_low_data = pd.concat([all_low_data,pd.DataFrame(ans.iloc[i,:]).T], ignore_index = True)
            all_high_data = pd.concat([all_high_data,pd.DataFrame(ans.iloc[j,:]).T], ignore_index = True)

# 王士博
ans = pd.read_csv(csv_path3)   
ans['quality'] = min_max_normalize(ans['quality'])
ans = ans[(ans['audibility2']!=0)]    
for i in range(0, len(ans)):
    for j in range(0, len(ans)):  
        if ans.iloc[j, 6] - ans.iloc[i, 6] > threshold:  #quality [i, 6]
            all_low_data = pd.concat([all_low_data,pd.DataFrame(ans.iloc[i,:]).T], ignore_index = True)
            all_high_data = pd.concat([all_high_data,pd.DataFrame(ans.iloc[j,:]).T], ignore_index = True)





# 定义对比学习模型
class Contrastive_model(nn.Module):
    def __init__(self,
                 n_fft: int = 2048,
                 ):
        super(Contrastive_model, self).__init__()

        # 我们所使用的一个线性层 (m, 1025) -> (m, 1025)
        # self.linear = nn.Linear(n_fft/2 + 1, n_fft/2 + 1)
        # self.linear = nn.Linear(1025, 1025)  # nn.Parameters

        self.a = torch.nn.Parameter(torch.empty(1025))  # 如何初始化 更换zeros
        self.b = torch.nn.Parameter(torch.empty(1025))
        self.set_parameters_a_b()


        self.audi = Audibility_new(
                    sample_rate = 16000,
                    win_length = 2048,
                    hop_length = 2048,
                    n_fft = 2048,
                    device = device,
                    mode='vmoat'
                    )
    def set_parameters_a_b(self) -> None: #初始化参数 a和b矩阵，借鉴linear的写法
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        # init.kaiming_uniform_(self.a, a=math.sqrt(5))

        fan_in = 1025   
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        init.uniform_(self.a, -bound, bound)
        init.uniform_(self.b, -bound, bound)

    def forward_once(self, wav_delta, ori_wav_x):
        
        power_normalized, theta = self.audi.before_audibility(wav_delta.unsqueeze(0).to(device), \
                        ori_wav_x)

        theta = self.a * theta + self.b     # theta = a * tmep  + b

        audibility2 = (power_normalized / theta).mean(dim=(1,2))
        return audibility2

    def forward(self, wav_delta1, ori_wav_x1, wav_delta2, ori_wav_x2):
        out1 = self.forward_once(wav_delta1, ori_wav_x1)
        out2 = self.forward_once(wav_delta2, ori_wav_x2)
        return out1, out2





###  训练模型
    
model = Contrastive_model().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
qtime = time.strftime("%Y-%m-%d(%H:%M:%S)", time.localtime())
num_epochs = 32
for epoch in tqdm(range(num_epochs)):
    # for i, ((image1, _), (image2, _)) in enumerate(zip(data1, data2)):
    # for idx in range(min(len(all_high_data), len(all_low_data))):
    for idx in range(min(len(all_high_data), len(all_low_data))):  #选取构造样本集中的1000个样本对


        ori_wav_x1, sr_x = torchaudio.load(all_low_data.iloc[idx,2])
        adv_wav_x1, sr_x = torchaudio.load(all_low_data.iloc[idx,3])
        ori_wav_x2, sr_x = torchaudio.load(all_high_data.iloc[idx,2])
        adv_wav_x2, sr_x = torchaudio.load(all_high_data.iloc[idx,3])



        ori_wav_x1 = wav_padding_2048(ori_wav_x1)
        adv_wav_x1 = wav_padding_2048(adv_wav_x1)
        ori_wav_x2 = wav_padding_2048(ori_wav_x2)
        adv_wav_x2 = wav_padding_2048(adv_wav_x2)



        wav_delta1 = adv_wav_x1-ori_wav_x1
        wav_delta2 = adv_wav_x2-ori_wav_x2

        optimizer.zero_grad()
        # 计算audibility
        out1, out2 = model(wav_delta1, ori_wav_x1, wav_delta2, ori_wav_x2)

        loss = torch.max(out1-out2, torch.tensor(0.0,requires_grad=True))   # quality小的减去大的，所以希望loss越小越好
        # loss = out1-out2
       
       
        loss.backward()
        optimizer.step()
        for param in model.parameters():
            param.data[param.data < 0] = 1e-6  # 0

        if idx % 10 == 0:
            print("Epoch: ", epoch+1, " Iteration: ", idx+1, " Loss: ", loss.item())




    # torch.save(model, '/home/vmoat/userstudy_wsbnew/contrastive_learn_fit3_model.pth')
    torch.save(model, '/home/vmoat/userstudy_wsbnew/contrastive_learn_fit3_model_norm5_all_epoch32.pth')

    # torch.save(model.state_dict(), '/home/vmoat/userstudy_wsbnew/contrastive_learn_fit3_dict_state3.pth')
    # torch.save(model.state_dict(), '/home/vmoat/userstudy_wsbnew/contrastive_learn_fit3_dict_state_norm5_all.pth')
    torch.save(model.state_dict(), f'/home/vmoat/userstudy_wsbnew/CL_model//contrastive_learn_fit3_dict_state_norm5_all_epoch{epoch}.pth')










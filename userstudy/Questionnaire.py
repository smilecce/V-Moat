import IPython.display as ipd
import ipywidgets as widgets
import json
import os, glob
import random 
import time

import pandas as pd

class questionnaire():
    def __init__(self):
        self.input_name()

    def input_name(self):
        self.namewidgets = widgets.Text(
                            value='',
                            placeholder='请输入你的姓名：',
                            description='姓名：',
                            disabled=False
                        )
        self.create_button = widgets.Button(
                            description='生成问卷',
                            disabled=False,
                            button_style='info', # 'success', 'info', 'warning', 'danger' or ''
                            tooltip='Click me',
                            icon='user' # (FontAwesome names without the `fa-` prefix)
                            )
        self.create_button.on_click(self.begin)
        ipd.display(self.namewidgets)
        ipd.display(self.create_button)
    
    def begin(self, x):
        self.inputstr = str.strip(self.namewidgets.value)
        if len(self.inputstr) != 0:
            self.create_button.disabled = True
            self.username = self.inputstr

            print('\n你好，{}！\n'.format(self.username))
            self.save_path = '/home/vmoat/userstudy_wsbnew/answers/'
            # self.statistics = '/home/vmoat/userstudy_wsbnew/statistics/adv_statistics_{}.csv'
            self.statistics = '/home/vmoat/userstudy_wsbnew/adv_statistics_11_level/adv_statistics_{}.csv'
            
            # self.statistics_files = [self.statistics.format(i) for i in range(0,12)]     #【s】：(1,6) --> (1,5)
            self.statistics_files = [self.statistics.format(i) for i in range(10,-1,-1)] 
            self.n_questions = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]     # 【s】： [20, 20, 20, 20, 20]  -->  [25, 25, 25, 25] 
            # self.n_questions = [2, 0, 0, 0, 0]

            self.create_questions()
            self.create_savebutton()    
        else:
            print('请输入姓名后再点击“生成问卷”。')

    def create_questions(self):
        pd_data = pd.DataFrame(None, columns=['id', 
                                    'ori_path', 
                                    'adv_path',
                                    # 'audibility1',
                                    'audibility2'])
        pd_data_temp = pd_data
        # random.seed(hash(self.username))
        random.seed(20231222)  #【s】改2:2023.12.22日

        for idx, csv_file in enumerate(self.statistics_files):
            this_data = pd.read_csv(csv_file, index_col=0)\
                                .sample(n=int(1), replace=False, random_state=20231222)      # 【s】改+1：  .sample(n=self.n_questions[idx]//2, replace=False)
            pd_data_temp = pd.concat([pd_data_temp, this_data])

        for idx, csv_file in enumerate(self.statistics_files):
            this_data = pd.read_csv(csv_file, index_col=0)\
                                .sample(n=int(self.n_questions[idx]*4/10), replace=False, random_state=20231222)      # 【s】改：  .sample(n=self.n_questions[idx]//2, replace=False)
            pd_data = pd.concat([pd_data, this_data])

            this_data = pd.read_csv('/home/vmoat/userstudy_wsbnew/statistics/clean_statistics.csv', index_col=0)\
                                .sample(n=int(self.n_questions[idx]*5/10), replace=False, random_state=20231222)      #  【s】改：  .sample(n=self.n_questions[idx]//2, replace=False)
            # random.shuffle(this_data)
            pd_data = pd.concat([pd_data, this_data])

        # pd_data = pd_data.sample(frac=1.0)    # 【s】改：是否进行全部打乱的选项，注释掉表示说不打乱了
        pd_data = pd.concat([pd_data_temp, pd_data])    #【s】 +
        self.comp_questions = []        
        
        for idx, (_, q) in enumerate(pd_data.iterrows()):
            self.comp_questions.append(compare_question(idx, q))

                
    def create_savebutton(self):
        self.button = widgets.Button(
                            description='保存',
                            disabled=False,
                            button_style='info', # 'success', 'info', 'warning', 'danger' or ''
                            tooltip='Click me',
                            icon='check' # (FontAwesome names without the `fa-` prefix)
                            )
        self.button.on_click(self.save_answers)
        ipd.display(self.button)
        
    def save_answers(self, x):
        self.answers = pd.DataFrame(None, columns=['id', 
                                    'ori_path', 
                                    'adv_path',
                                    # 'audibility1',
                                    'audibility2',
                                    'similarity',
                                    'quality'])
            
        for eq in self.comp_questions:
            self.answers = pd.concat([self.answers, eq.collect_ans()], ignore_index=True)
        
        self.qtime = time.strftime("%Y-%m-%d(%H:%M:%S)", time.localtime())
        self.ans_filename = self.save_path+str(self.username)+'_{}.csv'.format(self.qtime)
        self.answers.to_csv(self.ans_filename)
        self.qnaire = {
            'uid': self.username,
            'answer': self.ans_filename,
            'time': self.qtime
        }

        with open(self.save_path+str(self.username)+'_{}'.format(self.qtime), 'w') as f:
            json.dump(self.qnaire, f)
            f.close()
        print('问卷已保存，感谢参与，后会有期！')
            
        
def load_audio(path):
    import torchaudio
    audiodata, sr = torchaudio.load(path)
    audio = ipd.Audio(audiodata[:, :5*sr], rate = sr)
    display(audio)
    
def print_margin(index):
    print()
    print('-'*(100+10))
    print('|'+' '*49+'Question {:<2d}'.format(index)+' '*48+'|')
    #print('='*50+' Question {} '.format(index)+'='*50)
    print('-'*(100+10))
    print()
        
class compare_question():
    def __init__(self, index, q_row):
        self.q_row = q_row
        self.index = index
        
        print_margin(index)
                
        print('\n请听音频1：')
        load_audio(q_row['ori_path'])
        print('\n请听音频2：')
        load_audio(q_row['adv_path'])

        # question 1
        print('你认为这两个音频是一样的吗：')
        self.soundsimilarity = widgets.RadioButtons(
                                options=['same', 'different'],
                                layout={'width': 'max-content'},
                                description='Choices: ',
                                disabled=False
                            )
        ipd.display(self.soundsimilarity)
        print()

        print('请给音频中噪声（若有）的明显程度打分（0分：非常不明显，5分：非常明显）：')
        self.soundquality = widgets.widgets.FloatSlider(
                                value=0,
                                min=0,
                                max=5,
                                step=0.5,
                                description='Quality: ',
                                disabled=False,
                                continuous_update=False,
                                orientation='horizontal',
                                readout=True,
                                readout_format='.1f'
                            )
        ipd.display(self.soundquality) 
        print()

    def collect_ans(self):
        self.ans = pd.DataFrame({'id':[self.q_row['id']], 
                                'ori_path':[self.q_row['ori_path']], 
                                'adv_path':[self.q_row['adv_path']],
                                # 'audibility1':[self.q_row['audibility1']],
                                'audibility2':[self.q_row['audibility2']],
                                'similarity':[self.soundsimilarity.value],
                                'quality':[self.soundquality.value]},
                                )

        return self.ans
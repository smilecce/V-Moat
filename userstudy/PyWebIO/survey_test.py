from pywebio.input import *
from pywebio.output import *
from pywebio.pin import *
from pywebio import start_server
import pywebio_battery
import pandas as pd
import librosa
import random
import numpy as np
import soundfile as sf
import os
from pywebio.session import info as session_info
from pywebio.session import register_thread
import json
import time
import IPython.display as ipd
import torchaudio
from pydub import AudioSegment
import pydub
import threading
from threading import Thread
from time import sleep, ctime

def check_select(value):
    if value == -1:
        return "Please select an option"

class MyThread(threading.Thread):

    def __init__(self, func, args=()):
        super(MyThread, self).__init__()
        self.func = func
        self.args = args

    def run(self):
        self.result = self.func(*self.args)

    def get_result(self):
        try:
            return self.result   # 如果子线程不使用join方法，此处可能会报没有self.result的错误
        except Exception:
            return None

# import sounddevice as sd
# import soundfile as sf
class Survey():
    def __init__(self):
        # self.input_personal_info()
        input_part = Thread(target=self.input_personal_info)
        audition_part_1 = Thread(target=self.create_audition_audio)
        audition_part_2 = Thread(target=self.listening_audition_audio)
        test_part_1 = Thread(target=self.create_test_audios)
        test_part_2 = Thread(target=self.listening_test)
        # 注册pywebio的线程 注册线程，以便在线程内调用 PyWebIO 交互函数。 详细见：pywebio.session.register_thread
        register_thread(input_part)
        register_thread(audition_part_1)
        register_thread(audition_part_2)
        register_thread(test_part_1)
        register_thread(test_part_2)

        input_part.start()
        audition_part_1.start()
        test_part_1.start()

        input_part.join()
        audition_part_1.join()

        audition_part_2.start()
        audition_part_2.join()

        test_part_1.join()
        # test_part_2.start()
        # test_part_2.join()


    def load_audio(self, file_path):
        with open(file_path, 'rb') as file:
            audio_data = file.read()
            # print("Loaded audio:", file_path)
        return audio_data

    def load_audio_files_auditions(self):
        print('---Start---', 'time', ctime())
        for row in range(0,len(self.pd_data_audition)):
            # 创建并启动线程
            wav_path1 = self.pd_data_audition.iloc[row, 1]
            wav_path2 = self.pd_data_audition.iloc[row, 2]
            self.audition_audio_list_1.append(self.load_audio(wav_path1))
            self.audition_audio_list_2.append(self.load_audio(wav_path2))
        print('***End***', 'time', ctime())

    def load_audio_files_test(self):

        print('---Start---', 'time', ctime())
        for row in range(0,len(self.pd_data_test)):
            # print(row)
            # 第4列是options 如果是0的话，那么wav_path1就是ori，wav_path2就是adv
            if self.pd_data_test.iloc[row,4] ==0:
                wav_path1 = self.pd_data_test.iloc[row, 1]
                wav_path2 = self.pd_data_test.iloc[row, 2]
            elif  self.pd_data_test.iloc[row,4] ==1:
                wav_path1 = self.pd_data_test.iloc[row, 2]
                wav_path2 = self.pd_data_test.iloc[row, 1]

            self.test_audio_list_1.append(self.load_audio(wav_path1))
            self.test_audio_list_2.append(self.load_audio(wav_path2))
        print('***End***', 'time', ctime())
    def check_nationality(nationality):
        if(nationality == None):
            # toast(f'Questions not yet completed, please continue to answer', color='#2188ff',duration=3)
            return 'please select'
        else:
            return 'nomoral'       
                
    # 输入个人基本信息
    def input_personal_info(self):
        self.info = session_info #表示会话信息的对象，属性 其中包括很多内容，可以自动获取
        self.user_id = id(self.info['user_ip'])
        print(self.info)
        country = pd.read_csv('./COUNTRY_en.csv')
        print(country)
        country_list = [('Please choose', -1, True, True)]+ country['country'].tolist()
        print(country_list)
        # 问卷部分的 title
        put_markdown("# User Study")
        put_markdown("# Thank you for participating in this user study.")
        put_markdown('# Please enter your information before answering the questionaire:')
        # 必填
        self.start_time = time.time()   # 获取程序的开始时间

        info = input_group("User info",[
        # input('Name', required=True, name='name'),
        input('Worker ID', required=True, name='name'),
        # select('性别', options=[" ", "男", "女", "其他"], name='gender'),
        radio('Gender', required=True, options=[ "Male", "Female", "Other"], name='gender'),
        # select('性别', required=True, options=[("男",[False]), ("女",[False]), ("其他   ",[False])], name='gender'),
        # input('性别', required=True, placeholder = "请选择输入 男 女 其他 ", name='gender'),
        input('Age', required=True, name='age', type=NUMBER),
        # select('Country', required=True, options=country_list, name='nationality', validate=check_nationality)
        select('Country', required=True, options=country_list, name='nationality')
        ])

        # temp_flag = 1
        # print(info['nationality'])
        # while(temp_flag==1):
        #     if(info['nationality'] == None):
        #         toast(f'Questions not yet completed, please continue to answer', color='#2188ff',duration=3)       
        #     else:
        #         temp_flag=0
        #         break

        self.name = info['name']
        self.gender = info['gender']
        self.nationality = info['nationality']
        self.age = info['age']

        clear()
        # self.audition_part()
    def create_audition_audio(self):
        self.audition_audio_list_1 = []
        self.audition_audio_list_2 = []

        # # put_markdown("# 请听音频")
        # put_markdown("# 您好！")
        # put_markdown("本问卷旨在测试人耳分辨语音对抗样本的能力。\n请您选择**安静**环境，戴上**耳机**，完成本问卷。")
        # # put_markdown("> 请选择安静环境，带上耳机，完成本问卷")
        # put_markdown("首先是试听培训环节：")
        # put_markdown("在每一组音频中，第一段音频为干净音频，第二段音频含有不同大小的对抗噪声，**请您感受噪声的存在**，并尝试分辨两段音频。")

        self.save_path = '/mnt/data/old_home/strength/vmoat/userstudy_wsbnew/answers_under5s/'
        self.statistics = '/mnt/data/old_home/strength/vmoat/userstudy_wsbnew/adv_statistics_under5s/adv_statistics_{}.csv'
        self.statistics_files_audition = [self.statistics.format(i) for i in [10,6,3,1,0]] 
        self.n_questions_audition = [2, 2, 2, 2, 2] # 一共5个问题，然后下面乘以了1/2 

        self.pd_data_audition = pd.DataFrame(None, columns=['id', 
                                'ori_path', 
                                'adv_path',
                                'audibility2'])
        print('1'*1000)
        for idx, csv_file in enumerate(self.statistics_files_audition):
            this_data = pd.read_csv(csv_file, index_col=0)\
                                .sample(n=int(self.n_questions_audition[idx]*1/2), replace=False,random_state=42)      # 【s】改：  .sample(n=self.n_questions[idx]//2, replace=False)
            this_data['ori_path'] = this_data['ori_path'].str.replace('/home/vmoat', '/mnt/data/old_home/strength/vmoat', regex=False)
            this_data['adv_path'] = this_data['adv_path'].str.replace('/home/vmoat', '/mnt/data/old_home/strength/vmoat', regex=False)

            self.pd_data_audition = pd.concat([self.pd_data_audition, this_data])
        
        self.load_audio_files_auditions()
        # print('11111', self.audition_audio_list_1)

    def listening_audition_audio(self):
        
        # put_markdown("# 请听音频")
        put_markdown("# Hello!")
        put_markdown("The objective of this user study is to measure the ability of human auditory to distinguish audio adversarial examples (i.e., a kind of malicious noisy audios). Please pick a **quiet** environment and wear your **headphones** while filling the questionaire.")
        # put_markdown("> 请选择安静环境，带上耳机，完成本问卷")
        put_markdown("## First - Training Stage:")
        put_markdown("In each set of audios, the first one is clean, and the second one contains noise of different magnitudes. Please try your best to **feel the presence of noise** and distinguish the two audios.")

        for row in range(0,len(self.pd_data_audition)):
            # wav_path1 = self.pd_data_audition.iloc[row, 1]
            # wav_path2 = self.pd_data_audition.iloc[row, 2]
            put_markdown(f"# Question {row}")
            put_markdown("Listen to audio 1 :")
            # 下面的方式以字节的形式读取
            # with open(f"{wav_path1}", 'rb') as f1:
            #     audio_data_1 = f1.read()
            pywebio_battery.put_audio(self.audition_audio_list_1[row])

            put_markdown("Listen to audio 2 :")
            # with open(f"{wav_path2}", 'rb') as f2:
            #     audio_data_2 = f2.read()
            pywebio_battery.put_audio(self.audition_audio_list_2[row])
            tname = f"{self.pd_data_audition.iloc[row, 0]}"
            put_select(
                tname, 
                options=[("Please choose ...", -1, True, True), 
                        ("I can distinguish between two audios", 1, False, False), 
                        ("I can not distinguish between two audios", 0, False, False)], 
                )
            # put_select(
            #         tname, 
            #         options=[("请选择", -1, True, True), 
            #                 ("第一条有噪声", 1, False, False), 
            #                 ("第二条有噪声", 0, False, False), 
            #                 ("两段音频一样", 2, False, False)], 
            #         )


        put_markdown("# This is the end of the first part. Are you sure to enter the testing stage?")
        print(222222)
        # put_button(['确认开始'], onclick= self.create_test_audios)
        # button = put_button(['确认开始'], onclick= self.listening_test)
        button = put_button(['Confirm to start'], onclick= self.check_select_1)
        # button = put_button(['确认开始'], onclick= self.stop_clear)
        # button = actions('Confirm to delete file?', ['confirm', 'cancel'],
        #               help_text='Unrecoverable after file deletion')
        print(11111)
        
    # def stop_clear(self):
    #     clear()


    def check_select_1(self):
        print(1111111111111)
        # while(1):
        # while True:
        flag = 1
        # 完成作答后，再点击【确认开始】，即可开始
        for row in range(0,len(self.pd_data_audition)):
            tname = f"{self.pd_data_audition.iloc[row, 0]}"
            # pin_on_change(tname)
            print(pin[tname],type(pin[tname]))
            if pin[tname] == None: 
                # print(f'您有题目{row}尚未完成，请继续作答')
                toast(f'Question {row} is not yet completed. Please answer all questions.', color='#2188ff',duration=3)       
                flag = 0
                # pin_wait_change(tname, timeout=0)   #等待未作答的题目进行改变，完成作答后即可进行下一步
                # pin_on_change(tname, clear=True)
                # print(row)
                break

            # if not flag:
                # print('您有题目尚未完成，请继续作答')
                # toast('您有题目尚未完成，请继续作答', color='#2188ff',duration=4)
        if flag:
            self.audition_result = []
            print('ALL done')
            for row in range(0,len(self.pd_data_audition)):
                tname = f"{self.pd_data_audition.iloc[row, 0]}"
                self.audition_result.append(pin[tname])
                # print(pin[tname],type(pin[tname]))           
            self.listening_test()
            print('listening_test')
            # break

    # 试听部分
    # def audition_part_0(self):
        
    #     # put_markdown("# 请听音频")
    #     put_markdown("# 您好！")
    #     put_markdown("本问卷旨在测试人耳分辨语音对抗样本的能力。\n请您选择**安静**环境，戴上**耳机**，完成本问卷。")
    #     # put_markdown("> 请选择安静环境，带上耳机，完成本问卷")
    #     put_markdown("首先是试听培训环节：")
    #     put_markdown("在每一组音频中，第一段音频为干净音频，第二段音频含有不同大小的对抗噪声，**请您感受噪声的存在**，并尝试分辨两段音频。")

    #     self.save_path = '/home/vmoat/userstudy_wsbnew/answers_under5s/'
    #     self.statistics = '/home/vmoat/userstudy_wsbnew/adv_statistics_under5s/adv_statistics_{}.csv'
    #     self.statistics_files_audition = [self.statistics.format(i) for i in [10,6,3,1,0]] 
    #     self.n_questions_audition = [2, 2, 2, 2, 2]

    #     self.pd_data_audition = pd.DataFrame(None, columns=['id', 
    #                             'ori_path', 
    #                             'adv_path',
    #                             'audibility2'])
    #     for idx, csv_file in enumerate(self.statistics_files_audition):
    #         this_data = pd.read_csv(csv_file, index_col=0)\
    #                             .sample(n=int(self.n_questions_audition[idx]*1/2), replace=False)      # 【s】改：  .sample(n=self.n_questions[idx]//2, replace=False)
    #         self.pd_data_audition = pd.concat([self.pd_data_audition, this_data])
        
    #     for row in range(0,len(self.pd_data_audition)):
    #         wav_path1 = self.pd_data_audition.iloc[row, 1]
    #         wav_path2 = self.pd_data_audition.iloc[row, 2]
    #         put_markdown(f"# question {row}")
    #         put_markdown("请听音频1：")
    #         # 下面的方式以字节的形式读取
    #         with open(f"{wav_path1}", 'rb') as f1:
    #             audio_data_1 = f1.read()
    #         pywebio_battery.put_audio(audio_data_1)

    #         put_markdown("请听音频2：")
    #         with open(f"{wav_path2}", 'rb') as f2:
    #             audio_data_2 = f2.read()
    #         pywebio_battery.put_audio(audio_data_2)

    #     put_markdown("# 试听部分结束，是否确认开始听力测试？")
    #     print(222222)
    #     put_button(['确认开始'], onclick= self.create_test_audios)
    #     print(11111)

        




    def create_test_audios(self):
        self.test_audio_list_1 = []
        self.test_audio_list_2 = [] 
        print(1111111)
        # clear()
        # self.button0.safely_destruct()
        statistics_files = [self.statistics.format(i) for i in range(9,-1,-1)] 
        n_questions = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3]  # 30道题 十个level 每个level 3道题 这些都是关于adv的，还有15道题是clean的 

        # 创建pd_data  这样的 csv文件
        self.pd_data_test = pd.DataFrame(None, columns=['id', 
                                        'ori_path', 
                                        'adv_path',
                                        'audibility2'])

        # 前面的全是带adv的
        for idx, csv_file in enumerate(statistics_files):
            this_data = pd.read_csv(csv_file, index_col=0)\
                                .sample(n=int(n_questions[idx]*1), replace=False,random_state=42)      # 【s】改：  .sample(n=self.n_questions[idx]//2, replace=False)
            this_data['ori_path'] = this_data['ori_path'].str.replace('/home/vmoat', '/mnt/data/old_home/strength/vmoat', regex=False)
            this_data['adv_path'] = this_data['adv_path'].str.replace('/home/vmoat', '/mnt/data/old_home/strength/vmoat', regex=False)

            self.pd_data_test = pd.concat([self.pd_data_test, this_data])

        # 后面的全是clean的
        this_data = pd.read_csv('/mnt/data/old_home/strength/vmoat/userstudy_wsbnew/adv_statistics_under5s/clean_statistics.csv', index_col=0)\
                            .sample(n=int(15), replace=False,random_state=42)      #  【s】改：  .sample(n=self.n_questions[idx]//2, replace=False)
        self.pd_data_test = pd.concat([self.pd_data_test, this_data])
        random.seed(self.user_id)
        # 生成15个0和15个1的随机列表
        # option =0 表示 第一个是干净的   =1表示第二个是干净的
        random_list = [0] * 15 + [1] * 15 # 之后随机打乱表示15个第一个有噪声，15个第二个有噪声
        random.shuffle(random_list)     # 随机打乱列表的顺序
        random_list = random_list + [0] * 15  #加上clean的
        self.pd_data_test['option'] = random_list
        # 全部打乱
        self.pd_data_test = self.pd_data_test.sample(frac=1.0,random_state=42)    # 【s】改：是否进行全部打乱的选项，注释掉表示说不打乱了
        self.load_audio_files_test()
    #    
        # self.listening_test()
    def listening_test(self):
        clear()

        put_markdown("# Second - Testing Stage:")
        put_markdown("In each question, two audios might be the same or different. Please try your best to **distinguish the noisy audio**.")

        row_question = 0
        for row in range(0,len(self.pd_data_test)):
        # for row in range(0,5):
            if (row_question==25):
                put_markdown(f"# Question {row_question}")
                put_markdown("Listen to audio 1 :")
                # 下面的方式以字节的形式读取

                # pywebio_battery.put_audio(self.load_audio('/home/vmoat/vmoat_code/user_study_adv_audio/attack_verification_test-clean_under5s/model_weight_200_audi_61-70968-0044_adv_x/wav_x_adv_i9_4.9592632212365135.wav'))
                pywebio_battery.put_audio(self.load_audio('/mnt/data/old_home/strength/vmoat/vmoat_code/user_study_adv_audio/attack_verification_test-clean_under5s/model_weight_200_audi_8455-210777-0023_adv_x/wav_x_adv_i73_35514520.84725075.wav'))
                
                put_markdown("Listen to audio 2 :")

                pywebio_battery.put_audio(self.load_audio('/mnt/data/old_home/strength/vmoat/userstudy_wsbnew/LibriSpeech_under5s/test-clean/8455/210777/8455-210777-0023.flac'))
                put_markdown("> Please describe the two audios you hear")
                row_question = row_question + 1
                put_select('attention_1', options=[("Please choose ...", -1, True, True), ("The first audio has noise", 1, False, False), ("The second audio has noise.", 0, False, False), ("Two audios are the same", 2, False, False)])
                
                self.attention_1 = pin['attention_1']
            if (row_question==40):
                put_markdown(f"# Question {row_question}")
                put_markdown("Listen to audio 1 :")
                # 下面的方式以字节的形式读取

                # pywebio_battery.put_audio(self.load_audio('/home/vmoat/vmoat_code/user_study_adv_audio/attack_verification_test-clean_under5s/model_weight_200_audi_4446-2271-0006_adv_x/wav_x_adv_i2_33.89705945250451.wav'))
                pywebio_battery.put_audio(self.load_audio('/mnt/data/old_home/strength/vmoat/vmoat_code/user_study_adv_audio/attack_verification_test-clean_under5s/model_weight_200_audi_4970-29093-0021_adv_x/wav_x_adv_i52_16661969.268301377.wav'))
                
                put_markdown("Listen to audio 2 :")

                pywebio_battery.put_audio(self.load_audio('/mnt/data/old_home/strength/vmoat/userstudy_wsbnew/LibriSpeech_under5s/test-clean/4970/29093/4970-29093-0021.flac'))
                put_markdown("> Please describe the two audios you hear")
                row_question = row_question + 1
                put_select('attention_2', options=[("Please choose ...", -1, True, True), ("The first audio has noise", 1, False, False), ("The second audio has noise.", 0, False, False), ("Two audios are the same", 2, False, False)])
            
            put_markdown(f"# Question {row_question}")
            put_markdown("Listen to audio 1 :")
            # 下面的方式以字节的形式读取

            pywebio_battery.put_audio(self.test_audio_list_1[row])
            put_markdown("Listen to audio 2 :")

            pywebio_battery.put_audio(self.test_audio_list_2[row])


            # noise_result = select("请描述听到的两段音频", options=["第一条有噪声", "第二条有噪声", "两段音频一样"])
            # noise_result = checkbox("请描述听到的两段音频", ["第一条有噪声", "第二条有噪声", "两段音频一样"])
            put_markdown("> Please describe the two audios you hear")
            tname = f"{self.pd_data_test.iloc[row, 0]}"

            put_select(tname, options=[("Please choose ...", -1, True, True), ("The first audio has noise", 1, False, False), ("The second audio has noise.", 0, False, False), ("Two audios are the same", 2, False, False)])
            row_question = row_question + 1
        # confirm = actions('作答完毕请提交问卷', ['提交'])   #作为一个延时
        # input("是否提交")
        put_markdown("# You have finished the questionaire. Are you sure to submit?")

        # put_button(['确认提交'], onclick= self.save_ans)
        self.ans_submit = 0 #用来记录答案是否成功提交过，0表示没有成功提交过
        put_button(['Confirm to submit'], onclick=self.check_select_2)

        # put_button(['确认提交'], onclick= self.save_ans)
        print(3333333)
        # clear()
    def check_select_2(self):
        print(1111111111111)
        # while(1):
        # while True:
            # changed = pin_wait_change()
        flag = 1
        for row in range(0,len(self.pd_data_test)):
        # for row in range(0,5):

            tname = f"{self.pd_data_test.iloc[row, 0]}"
            # print(pin[tname],type(pin[tname]))
            if pin[tname] == None: 
                # print(f'您有题目{row}尚未完成，请继续作答')
                toast(f'Question {row} is not yet completed. Please answer all questions.', color='#2188ff',duration=3)       
                flag = 0
                pin_wait_change(tname, timeout=0)
                # pin_on_change(tname, clear=True)
                # print(row)
                break
        
            # if flag:
            #     break
            # if not flag:
                # print('您有题目尚未完成，请继续作答')
                # toast('您有题目尚未完成，请继续作答', color='#2188ff',duration=4)
        if (flag==1) and (self.ans_submit==0):
            self.ans_submit = 1 #用来记录答案是否成功提交过，这个来保证只能成功提交一次
            print('save_ans')
            self.end_time = time.time()   #程序的结束时间
            # toast(f'问卷已全部完成，感谢您的参与!', color='#2188ff',duration=2) 
            self.attention_2 = pin['attention_2']           
            self.attention_1 = pin['attention_1']           

            # print(self.attention_1)
            # print(type(self.attention_1))
            # print(pin['attention_1'])
            # print(type(pin['attention_1']))
            self.save_ans()
            # break
    # def listening_test_0(self):
    #     for row in range(0,len(self.pd_data_test)):
    # # for row in range(0,2):
    #         if self.pd_data_test.iloc[row,4] ==0:
    #             wav_path1 = self.pd_data_test.iloc[row, 1]
    #             wav_path2 = self.pd_data_test.iloc[row, 2]
    #         elif  self.pd_data_test.iloc[row,4] ==1:
    #             wav_path1 = self.pd_data_test.iloc[row, 2]
    #             wav_path2 = self.pd_data_test.iloc[row, 1]

    #         put_markdown(f"# question {row}")
    #         put_markdown("请听音频1：")
    #         # 下面的方式以字节的形式读取
    #         with open(f"{wav_path1}", 'rb') as f1:
    #             audio_data_1 = f1.read()
    #         pywebio_battery.put_audio(audio_data_1)
    #         put_markdown("请听音频2：")
    #         with open(f"{wav_path2}", 'rb') as f2:
    #             audio_data_2 = f2.read()
    #         pywebio_battery.put_audio(audio_data_2)


    #         # noise_result = select("请描述听到的两段音频", options=["第一条有噪声", "第二条有噪声", "两段音频一样"])
    #         # noise_result = checkbox("请描述听到的两段音频", ["第一条有噪声", "第二条有噪声", "两段音频一样"])
    #         put_markdown("> 请描述听到的两段音频")
    #         tname = f"{self.pd_data_test.iloc[row, 0]}"
    #         put_select(tname, options=[("第一条有噪声", 1), ("第二条有噪声", 0), ("两段音频一样", 2)])

    #     # confirm = actions('作答完毕请提交问卷', ['提交'])   #作为一个延时
    #     # input("是否提交")
    #     put_markdown("# 问卷已完成，是否确认提交？")
    #     # self.button = put_button(['确认提交'], onclick=lambda: self.save_ans())
    #     put_button(['确认提交'], onclick= self.save_ans)
    #     print(3333333)
    #     # clear()
    def save_ans(self):
        print(3333333)
        # clear()
        self.select_result = []
        print('-'*100)
        for row in range(0,len(self.pd_data_test)):
            tname = f"{self.pd_data_test.iloc[row, 0]}"
            # print(pin[tname],type(pin[tname]))
            self.select_result.append(pin[tname])
        clear()
        random.seed(self.user_id)
        self.attention = 'unfocused'
        if((self.attention_1==1) and (self.attention_2==1)):
            self.attention = 'focused'
        rand_name = random.randint(1,100000)
        qtime = time.strftime("%Y-%m-%d(%H:%M:%S)", time.localtime())
        self.run_time = self.end_time - self.start_time
        folder = f"{self.save_path}{self.attention}_{self.name}_{self.gender}_{self.age}_{self.nationality}_{qtime}_{self.run_time:.3f}s_{rand_name:05d}"
        if not os.path.exists(folder):
            os.makedirs(folder)


        pd.DataFrame(self.audition_result).to_csv(f'{folder}/audition_result.csv')

        self.pd_ans = self.pd_data_test
        self.pd_ans['select_result'] = self.select_result
        

        # self.pd_ans.to_csv(f"{self.save_path}{self.name}_{self.gender}_{self.age}_{self.nationality}_{qtime}{rand_name:05d}.csv")

        self.pd_ans.to_csv(f"{folder}/test_result.csv")

        # 显示感谢信息
        put_markdown("# Thank you!")
        put_markdown("## survey code: "+str(hash(self.name))[:10])
if __name__ == '__main__':
    start_server(Survey, port=9090,debug=True)
















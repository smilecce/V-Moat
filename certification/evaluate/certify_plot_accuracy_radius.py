import argparse
import os, glob
from tqdm import tqdm

import pandas as pd
import torch
import torchaudio
import librosa

from model.sb_interfaces import Verification, Verification_CMK
from model.sb_interfaces import Classification

from certification.vmoat import Vmoat
root_path = '/mnt/data/old_home/strength/vmoat'


# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  #【s】：自己加的，用来设置多gpu运行
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='Certify many examples')
# parser.add_argument("dataset", choices=DATASETS, help="which dataset")
# parser.add_argument("base_classifier", type=str, help="path to saved pytorch model of base classifier")
parser.add_argument("--model_name", type=str, help=" path to saved pytorch model of base classifier") #defult 0.001
parser.add_argument("--sigma", type=float, default=0.25, help="noise hyperparameter")
# parser.add_argument("outfile", type=str, help="output file")
parser.add_argument("--batch", type=int, default=120, help="batch size")
# parser.add_argument("--skip", type=int, default=1, help="how many examples to skip")
# parser.add_argument("--max", type=int, default=-1, help="stop after this many examples")
# parser.add_argument("--split", choices=["train", "test"], default="test", help="train or test set")
parser.add_argument("--N0", type=int, default=100)
parser.add_argument("--N", type=int, default=100000, help="number of samples to use")     # 默认的n是100000
# parser.add_argument("--N0", type=int, default=1)
# parser.add_argument("--N", type=int, default=1, help="number of samples to use")     # 默认的n是100000
parser.add_argument("--alpha", type=float, default=0.001, help="failure probability") #defult 0.001
parser.add_argument("--model_src", type=str, help=" path of the pytorch model of base classifier") #defult 0.001

args = parser.parse_args()

class blank_classifier(torch.nn.Module):
    def __init__(self, device) -> None:
        super().__init__()
        self.device = device


def generate_trials(n: int = 15, filename: str= './test_trials_1200_Nf24.csv'):
    import random

    librispeech_path = '/home/vmoat/LibriSpeech/test-clean/'
    all_wavs = glob.glob(os.path.join(librispeech_path, '*/*/*.flac'))
    spks = os.listdir(librispeech_path)

    trials = pd.DataFrame([], columns=['enroll', 'trial', 'groundtruth'])

    for spk in tqdm(spks):
        wavs = glob.glob(os.path.join(librispeech_path, spk+'/*/*.flac'))
        if len(wavs) >= 2*n:
            num = n
        else:
            num = len(wavs) // 2

        same_enroll_wavs = wavs[:num]
        same_trial_wavs = [] 
        diff_enroll_wavs = wavs[num:2*num]
        diff_trial_wavs = []

        this_same_samples = random.sample(wavs, 2*num)
        this_diff_samples = random.sample(all_wavs, 2*num)
        
        # collect samples from the same speaker
        for wav in this_same_samples:
            if librosa.get_duration(filename=wav) < 24 * 2048 / 16000:
                pass
            else:
                same_trial_wavs.append(wav)
                if len(same_trial_wavs) == num:
                    break

        # collect samples from different speakers
        for wav in this_diff_samples:
            if wav.split('/')[-3] == spk:
                pass
            elif librosa.get_duration(filename=wav) < 24 * 2048 / 16000:
                pass
            else:
                diff_trial_wavs.append(wav)
                if len(diff_trial_wavs) == num:
                    break

        assert len(diff_trial_wavs) == num
        tmp_dict = {
            'enroll': same_enroll_wavs + diff_enroll_wavs,
            'trial': same_trial_wavs + diff_trial_wavs,
            'groundtruth': [True]*num + [False]*num
        }
        # print(len(tmp_dict['enroll']), len(tmp_dict['trial']), len(tmp_dict['groundtruth']))
        trials = pd.concat([trials, pd.DataFrame(tmp_dict)])

    trials.to_csv(filename)
    exit()


if __name__ == "__main__":
    # generate_trials()

    # model_name = 'FT33Sigmafix_10'
    # sigma = 10
    model_name = args.model_name
    sigma = args.sigma
    csv_name = f'certify_plot_cnaudio_folder/{model_name}_sigma_{int(sigma) }_N_{args.N}_a_{args.alpha}_Nf24.csv'

    assert not os.path.exists(csv_name), 'The CSV file already exists. You are attempting to OVERRIDE existing file.'
    
    print(csv_name)

    verification_pd = pd.read_csv('./purify_audio_trial_cn1000.csv')
    # import ipdb;ipdb.set_trace()
    verification_pd['enroll'] = verification_pd['enroll'].str.replace('/home/vmoat', root_path, regex=False)
    verification_pd['trial'] = verification_pd['trial'].str.replace('/home/vmoat', root_path, regex=False)

    enroll_wavs = verification_pd['enroll']
    trial_wavs = verification_pd['trial']
    groundtruth = verification_pd['groundtruth']

    # 实例化一个verification for verfiy
    # verification = Verification.from_hparams(
    #                             source="speechbrain/spkrec-ecapa-voxceleb",
    #                             savedir="pretrained_models/spkrec-ecapa-voxceleb",
    #                             run_opts = {
    #                                 "device": device,
    #                                 "data_parallel_backend": True,
    #                             })
    # model_src = '/home/vmoat/vmoat_code/model/trained_models/CKPT+2023-05-15+09-13-38+00'
    # model_src = '/home/vmoat/vmoat_code/model/trained_models/CKPT+2023-07-08+21-36-24+00'
    model_src = args.model_src
    print('model_src】】】', model_src)
    hparams_src = 'test_ecapa_tdnn.yaml'
    # savedir = '/home/vmoat/vmoat_code/model/trained_models/CKPT+2023-07-08+21-36-24+00'
    savedir = model_src
    verification = Verification.from_hparams(
                                source=model_src,
                                hparams_file=hparams_src,
                                savedir=savedir,
                                run_opts = {
                                    "device": device,
                                    "data_parallel_backend": True,
                                })
    # model_src = '/home/vmoat/vmoat_code/model/trained_models/CKPT+2023-05-24+14-54-03+00'
    # hparams_src = 'test_ecapa_tdnn.yaml'
    # savedir = '/home/vmoat/vmoat_code/model/trained_models/CKPT+2023-05-24+14-54-03+00'
    # verification = Verification_CMK.from_hparams(
    #                             source=model_src,
    #                             hparams_file=hparams_src,
    #                             savedir=savedir,
    #                             run_opts = {
    #                                 "device": device,
    #                                 "data_parallel_backend": True,
    #                             })
    verification.eval()
    smoothed_verification = Vmoat(verification, num_classes = 2, sigma = sigma)

    results = {
        'prediction': [],
        'radius': [],
        'right': []
    }

    for idx in tqdm(range(len(enroll_wavs))):
    # for idx in tqdm(range(477, len(enroll_wavs))):
        # enroll a wav for verfication
        wav, _ = torchaudio.load(enroll_wavs[idx])
        verification.enroll(wav.to(device))
        
        wav_x, _ = torchaudio.load(trial_wavs[idx])

        if wav_x.size(1) < 24*2048:
            print(f"\n警告：跳过第 {idx} 对音频，因为时长太短。")
            continue # 跳过当前循环，继续处理下一对

        assert wav_x.size(1) >= 24*2048

        wav_x = wav_x[:, -24*2048:]
        # torch.manual_seed(123)
        prediction, radius = smoothed_verification.certify(wav_x.to(device), 
                                            args.N0, args.N, args.alpha, args.batch)

        results['prediction'].append(prediction)
        results['radius'].append(radius)

        if prediction == -1:
            results['right'].append(-1)
        elif (prediction == 0) == groundtruth[idx]:
            results['right'].append(True)
        else:
            results['right'].append(False)

        # print(prediction, radius, results['right'][-1], enroll_wavs[idx], trial_wavs[idx])

        results1 = pd.DataFrame(results)
        results1.to_csv(csv_name)

    exit()

    # enroll_speaker_path = glob.glob(r"/home/vmoat/LibriSpeech/test-clean/61/*/*.flac")[0]
    # enrolled_speaker_wav, sr = torchaudio.load(enroll_speaker_path)

    # 50 same speakers and 50 different speakers for verfiy
    # same_speaker_paths = glob.glob(r"/home/vmoat/LibriSpeech/test-clean/61/*/*.flac")[1:51]
    # diff_speaker_paths = glob.glob(r"/home/vmoat/LibriSpeech/test-clean/121/*/*.flac")[0:50]

    # verfiy_speaker_paths = same_speaker_paths + diff_speaker_paths

    x_radius = [x / 10 for x in range(0, 11)]
    bingo_prediction_count_list = [0]*10  #len should be equal to len(x_radius)
    bingo_prediction = [0] * 50 + [1] * 50
    accuracy_list = []
    radius_list = []
    prediction_list = []
    # for j in tqdm(range(0,10)):
    #     id_x = 0
    for path_i in tqdm(verfiy_speaker_paths):

        path3 = path_i
        wav3, sr = torchaudio.load(path3)



        wav_x = wav3
        # call the forward function of the verfication，return the similarity of the 'input_wav' and the 'enrolled_wav' 
        # return [similarity threshold]   也就是相似对阈值的对比，当相似度大于阈值时，那么可以认为wav1和enrolled_wav是同一个说话者。
        # p = verification(wav_x.to(device))        #这里就是调用forward函数  
        # print('【similarity threshold】', p)

        # create the smooothed classifier g
        smoothed_verification = Vmoat(verification, num_classes = 2, sigma = 0.75)
        # smoothed_verification = Smooth(verification, num_classes = 2, sigma = 0.15)

        # predictition_true  就是实例化的verfication的输出，会判断出两个是否相等。
        # smoothed_predictions    
        prediction, radius = smoothed_verification.certify(wav_x.to(device), args.N0, args.N, args.alpha, args.batch)
        # accuracy = smoothed_verification.accuracy
        # print('【accuracy】', accuracy)
        print('path_i', path_i)
        print('【prediction, radius】', prediction, radius)    #【prediction， radius】 0 0.675458239509314
        radius_list.append(radius)
        prediction_list.append(prediction)

            # 1.radius >x_radius
            # 2.verfiy correct
            # if radius > x_radius[j]:
            #     if prediction == bingo_prediction[id_x]:
            #         bingo_prediction_count_list[j] = bingo_prediction_count_list[j] + 1

            # id_x = id_x + 1
        # accuracy_list.append(bingo_prediction_count_list[j]/100)
        # print('bingo_prediction_count_list[j]/100', bingo_prediction_count_list[j]/100)
    print('【radius_list】', radius_list)
    print('【prediction_list】' ,prediction_list)
    radius_df = pd.DataFrame(radius_list, index=verfiy_speaker_paths ,columns=['radius'])
    # radius_df.to_csv('radius.csv')
    prediction_df = pd.DataFrame(prediction_list, index=verfiy_speaker_paths ,columns=['prediction'])
    # prediction_df.to_csv('prediction.csv')
    radius_prediction = pd.concat([radius_df,prediction_df], axis=1)
    # radius_prediction.to_csv('finetune_radius_prediction_sigma0.75.csv')
    radius_prediction.to_csv('radius_prediction_sigma0.75.csv')
    # accuracy_df = pd.DataFrame(accuracy_list)
    # print('【bingo_prediction_count_list】', bingo_prediction_count_list)
    # accuracy_df.to_csv('accuracy_3.csv')





            # if prediction == 0:
            #     print('verify result: same speaker')
            # else:
            #     print('verify result: different speaker')

        # accuracy_list.append(accuracy)
        # radius_list.append(radius)

    # df_radius_accuracy = pd.concat([pd.DataFrame(accuracy_list), pd.DataFrame(radius_list)], axis=1)
    # print(df_radius_accuracy)
    # df_radius_accuracy.to_csv('radius_accracy_speaker_121_sigma_1.csv')
    
exit()































########################################################################
## classification
########################################################################
# enroll_list_dict = [
#                     # {'wav_path':'/home/vmoat/shibo/Code/paper1_random_smooth/smoothing-master/code/spk1_snt1.wav','spk_id':'spk1'},
#                     {'wav_path':'/home/vmoat/shibo/Code/paper1_random_smooth/smoothing-master/code/spk1_snt2.wav','spk_id':'spk1'},
#                     {'wav_path':'/home/vmoat/shibo/Code/paper1_random_smooth/smoothing-master/code/spk2_snt2.wav','spk_id':'spk2'},
#                     {'wav_path':'/home/vmoat/shibo/Code/paper1_random_smooth/smoothing-master/code/spk19-198-0000.flac','spk_id':'spk19'},
#                     # {'wav_path':'/home/vmoat/shibo/Code/paper1_random_smooth/smoothing-master/code/spk19-198-0001.flac','spk_id':'spk19'}
#                     ]
enroll_list_dict = [
                    {'wav_path':'/home/vmoat/LibriSpeech/test-clean/61/70968/61-70968-0000.flac','spk_id':'61'},
                    {'wav_path':'/home/vmoat/LibriSpeech/test-clean/121/121726/121-121726-0000.flac','spk_id':'121'},
                    {'wav_path':'/home/vmoat/LibriSpeech/test-clean/237/126133/237-126133-0000.flac','spk_id':'237'},
                    ]
# SPK_id = ['spk1','spk2','spk19']

classification = Classification.from_hparams(
                    source="speechbrain/spkrec-ecapa-voxceleb",
                    run_opts = {
                        "device": device,
                        "data_parallel_backend": False,
                    })

classification.enroll_many(enroll_list_dict)        #这里就是调用forward函数  
p = classification(wav1.to(device))
print('结果', p)

# smoothed classification


num_classes = classification.num_classes()
print('num_classes', num_classes)
# smoothed_classification = Smooth(classification, num_classes = num_classes, sigma = 0.05)  #【before】
smoothed_classification = Vmoat(classification, num_classes = num_classes, sigma = 1e4)     #【vmoat】

prediction, radius = smoothed_classification.certify(wav2.to(device), args.N0, args.N, args.alpha, args.batch)

print('【prediction, radius】', prediction, radius)    #【prediction， radius】 0 0.675458239509314
# print(classification.spkid_list)
print('classify result most_similar_spk_id:', classification.index_decode_2spk_id(prediction))













   
   
    #先不smooth
    # smoothed_classifier = Smooth(base_classifier, get_num_classes(dataset), args.sigma)
    # f = open(args.outfile, 'w')
    # print("idx\tlabel\tpredict\tradius\tcorrect\ttime", file=f, flush=True)
    # for i in range(len(dataset)):

    #         # only certify every args.skip examples, and stop after args.max examples
    #     if i % args.skip != 0:
    #             continue
    #     if i == args.max:
    #         break

    #     (x, label) = dataset[i]

    #     before_time = time()                                 #【s】:开始certify时间
    #         # certify the prediction of g around x
    #     x = x.cuda()                                         #【s】
    #     prediction, radius = base_classifier.certify(x, args.N0, args.N, args.alpha, args.batch)

    #     # prediction, radius = smoothed_classifier.certify(x, args.N0, args.N, args.alpha, args.batch)
    #     after_time = time()                                  #【s】：certify运行结束的结束时间
    #     correct = int(prediction == label)                   #【s】correct=0或者1     pred=label等于的话就是1

    #     time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))
    #     print("{}\t{}\t{}\t{:.3}\t{}\t{}".format(
    #         i, label, prediction, radius, correct, time_elapsed), file=f, flush=True)

    # f.close()













    # prepare output file
    # f = open(args.outfile, 'w')
    # print("idx\tlabel\tpredict\tradius\tcorrect\ttime", file=f, flush=True)

    # # iterate through the dataset
    # # dataset = get_dataset(args.dataset, args.split)

    # ###【s】：这里开始进行certify， 对每个id进行判断，但是会进行
    # for i in range(len(dataset)):

    #     # only certify every args.skip examples, and stop after args.max examples
    #     if i % args.skip != 0:
    #         continue
    #     if i == args.max:
    #         break

    #     (x, label) = dataset[i]

    #     before_time = time()                                 #【s】:开始certify时间
    #     # certify the prediction of g around x
    #     x = x.cuda()                                         #【s】
    #     prediction, radius = smoothed_classifier.certify(x, args.N0, args.N, args.alpha, args.batch)
    #     after_time = time()                                  #【s】：certify运行结束的结束时间
    #     correct = int(prediction == label)                   #【s】correct=0或者1     pred=label等于的话就是1

    #     time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))
    #     print("{}\t{}\t{}\t{:.3}\t{}\t{}".format(
    #         i, label, prediction, radius, correct, time_elapsed), file=f, flush=True)

    # f.close()





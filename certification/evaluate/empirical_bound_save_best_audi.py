import torch
from torch import nn
import torchaudio
from tqdm import tqdm
from certification.masker import Masker
import numpy as np
import os
import random
from glob import glob
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import seaborn as sns


# os.environ["CUDA_VISIBLE_DEVICES"] = "0,2,3"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser(description='Certify many examples')
# parser.add_argument("dataset", choices=DATASETS, help="which dataset")
# parser.add_argument("base_classifier", type=str, help="path to saved pytorch model of base classifier")
# parser.add_argument("sigma", type=float, default=0.25, help="noise hyperparameter")
# parser.add_argument("outfile", type=str, help="output file")
parser.add_argument("--batch", type=int, default=100, help="batch size")
# parser.add_argument("--skip", type=int, default=1, help="how many examples to skip")
# parser.add_argument("--max", type=int, default=-1, help="stop after this many examples")
# parser.add_argument("--split", choices=["train", "test"], default="test", help="train or test set")
parser.add_argument("--N0", type=int, default=100)
parser.add_argument("--N", type=int, default=100, help="number of samples to use")     # 默认的n是100000
parser.add_argument("--alpha", type=float, default=0.001, help="failure probability")
args = parser.parse_args()

EPS = np.finfo(np.float32).eps

def pgd_attack(model: torch.nn.Module, 
               wav_x: torch.Tensor, 
               labels: torch.Tensor, 
               eps: float=0.1, 
               alpha: float=0.01, 
               iters: int=40, 
               device='cuda'):
    ''' The naive PGD attack on audio models.

    Arguments
    ---------
    model : torch.nn.Module
        The model to attack.
    wav_x : torch.Tensor
        The original wav of size (1, n_points)
    labels : torch.Tensor
        The target label, e.g., torch.as_tensor([1])
    eps : float
        The L_infty norm constraint. Default: 0.1. 
    alpha : float
        The learning rate. Default: 0.01.
    iters : int
        The epochs of optimizaiton. Default: 40. 
    device : torch.device
        Default: 'cuda'.
    
    Returns
    -------
    wav_x : torch.Tensor
        The adversarial example audio.
    '''
    wav_x = wav_x.to(device)
    labels = labels.to(device)
    ce_loss = nn.CrossEntropyLoss()
        
    ori_wav_x = wav_x.data
        
    for _ in tqdm(range(iters)):    
        wav_x.requires_grad = True
        outputs = model(wav_x)

        model.zero_grad()
        cost = ce_loss(outputs, labels).to(device)
        cost.backward()

        adv_wav_x = wav_x - alpha * wav_x.grad.sign()
        eta = torch.clamp(adv_wav_x - ori_wav_x, min=-eps, max=eps)
        wav_x = torch.clamp(ori_wav_x + eta, min=-1, max=1).detach_()
            
    return wav_x


def pgd_attack_psy_smooth_multiclass(model, wav_x, target_speaker_idx, eps=0.3, alpha=0.05, iters=100, device='cuda'):
    ''' 更激进的攻击版本 - 针对高鲁棒性模型 '''
   
    wav_x = wav_x.to(device)
    target_label = torch.tensor([target_speaker_idx], device=device)

    msker = Masker(
        device = 'cuda',
        win_length = 2048,
        hop_length = 2048,
        n_fft = 2048,
        sample_rate = 16000,
        )
    theta, original_max_psd = msker._compute_masking_threshold(
                        wav_x[0,:].cpu().numpy(), mode="vmoat")
    theta += EPS
    original_max_psd += EPS

    ori_wav_x = wav_x.data
    wav_x = wav_x.to(device)

    best_wav_x = wav_x.data
    best_audi = np.inf
    
    # 修改1：更宽松的置信度阈值
    upper_confidence = 0.3  # 从0.5降到0.3，更容易触发攻击
    
    # 修改2：动态调整心理声学权重
    coeff = 1e-3  # 开始时更小的心理声学权重，优先攻击成功

    pbar = tqdm(range(iters))
    for i in pbar:    
        wav_x.requires_grad = True
        outputs = model.predict_to_be_attack(wav_x, n=100, batch_size=100)

        model.zero_grad()
        
        # 修改3：使用不同的损失函数
        # 不仅要降低真实类别置信度，还要提高其他类别置信度
        true_class_conf = outputs[:, target_speaker_idx]
        
        # 找到最高的非真实类别置信度
        outputs_masked = outputs.clone()
        outputs_masked[:, target_speaker_idx] = -999  # 屏蔽真实类别
        max_other_conf = outputs_masked.max(dim=1)[0]
        
        # 新的损失：希望 other_class_conf > true_class_conf + margin
        margin = 0.1
        ce_cost = torch.relu(true_class_conf - max_other_conf + margin)
        
        psy_cost = msker.batch_forward_2nd_stage(
                        local_delta_rescale=wav_x-ori_wav_x,
                        theta_batch=torch.as_tensor(theta).to(device).T,
                        original_max_psd_batch=torch.as_tensor(original_max_psd).to(device),
                        mode="vmoat"
                        )
        
        pbar.set_description(f"ce: {ce_cost.item():.5f} psy: {psy_cost.item():.8f} conf_gap: {(max_other_conf-true_class_conf).item():.3f}")
        cost = ce_cost + coeff * psy_cost
        cost.backward()

        wav_x.grad[wav_x.grad.isnan()] = 0
        adv_wav_x = wav_x - alpha * wav_x.grad
        eta = torch.clamp(adv_wav_x - ori_wav_x, min=-eps, max=eps)
        wav_x = torch.clamp(ori_wav_x + eta, min=-1, max=1).detach_()

        # 修改4：更宽松的成功判定条件
        with torch.no_grad():
            outputs_check = model.predict_to_be_attack(wav_x, n=100, batch_size=100)
            predicted_class = outputs_check.argmax().item()
            confidence_gap = outputs_check[:, predicted_class] - outputs_check[:, target_speaker_idx]
            
            # 成功条件：预测类别不同 OR 置信度差距足够大
            attack_success = (predicted_class != target_speaker_idx) or (confidence_gap > 0.05)
            
        if attack_success:
            print(f'ATTACK SUCCESS! True: {target_speaker_idx}, Predicted: {predicted_class}, Gap: {confidence_gap.item():.4f}, Audibility: {psy_cost.item():.8f}')
            coeff = 1e-5  # 攻击成功后，大幅降低心理声学权重，专注优化音质
            
            if psy_cost < best_audi:
                best_audi = psy_cost.item()
                best_wav_x = wav_x.data
                print(f'UPDATE BEST AUDIBILITY: {psy_cost.item():.8f}')

            # if psy_cost.item() < 1e-6:  # 非常小的扰动就停止
            if psy_cost.item() < 100000:  # 非常小的扰动就停止

                return best_wav_x, best_audi
        else:
            # 修改5：动态调整参数
            if i > 0 and i % 20 == 0:  # 每20轮检查一次
                if ce_cost.item() > 0.1:  # 如果损失还很大
                    upper_confidence = max(upper_confidence - 0.02, 0.1)  # 进一步放宽条件
                    alpha = min(alpha * 1.1, 0.1)  # 增大学习率
                    print(f'Adjusting: upper_conf={upper_confidence:.3f}, alpha={alpha:.4f}')
            
    return best_wav_x, best_audi

def extract_speaker_id_from_path(file_path):
    """Extract speaker ID from LibriSpeech file path.
    
    LibriSpeech format: /path/to/speaker_id/chapter_id/speaker_id-chapter_id-utterance_id.flac
    """
    parts = file_path.split('/')
    # Find the speaker ID (should be numeric and appear twice in the path)
    for part in parts:
        if part.isdigit():
            return part
    return None


def validate_and_clean_csv(df):
    """Validate DataFrame and remove non-existent audio files.
    
    Arguments
    ---------
    df : pd.DataFrame
        DataFrame with audio_path column to validate
        
    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with only existing files
    """
    valid_rows = []
    
    print(f"Validating {len(df)} audio files...")
    for idx, row in df.iterrows():
        if os.path.exists(row['audio_path']):
            valid_rows.append(row)
        else:
            print(f"File not found, removing: {row['audio_path']}")
    
    cleaned_df = pd.DataFrame(valid_rows)
    print(f"Kept {len(cleaned_df)} valid files out of {len(df)} total files")
    
    return cleaned_df


def create_multiclass_csv_from_pairs(pairs_csv_path, output_csv_path):
    """Convert pairs CSV to multi-class CSV format.
    
    Arguments
    ---------
    pairs_csv_path : str
        Path to the original pairs CSV
    output_csv_path : str
        Path to save the new multi-class CSV
    """
    df = pd.read_csv(pairs_csv_path)
    
    # Extract all unique audio files and their speaker IDs
    all_files = []
    
    for _, row in df.iterrows():
        enroll_spk = extract_speaker_id_from_path(row['enroll'])
        trial_spk = extract_speaker_id_from_path(row['trial'])
        
        all_files.append({'audio_path': row['enroll'], 'speaker_id': enroll_spk})
        all_files.append({'audio_path': row['trial'], 'speaker_id': trial_spk})
    
    # Remove duplicates and create new DataFrame
    multiclass_df = pd.DataFrame(all_files).drop_duplicates()
    multiclass_df.to_csv(output_csv_path, index=False)
    
    return multiclass_df


def plot_empirical_robust_accuracy(results_df, output_dir, max_radius=5.0, num_points=50):
    """Plot empirical robust accuracy vs attack radius.
    
    Arguments
    ---------
    results_df : pd.DataFrame
        DataFrame containing results with columns: 'correct', 'best_audi', 'radius'
    output_dir : str
        Directory to save the plot
    max_radius : float
        Maximum radius to plot
    num_points : int
        Number of points to evaluate
    """
    
    # Create radius points to evaluate
    radius_points = np.linspace(0, max_radius, num_points)
    robust_accuracies = []
    
    # Filter out failed attacks (best_audi == inf)
    valid_results = results_df[results_df['best_audi'] != np.inf].copy()
    
    if len(valid_results) == 0:
        print("Warning: No successful attacks found, using certification radius only")
        valid_results = results_df.copy()
        valid_results['attack_radius'] = valid_results['radius']  # Use certification radius
    else:
        valid_results['attack_radius'] = valid_results['best_audi']  # Use attack radius
    
    total_samples = len(valid_results)
    
    for r in radius_points:
        # Count samples that are:
        # 1. Correctly classified (correct == True)
        # 2. Have attack radius > r (meaning they're robust to attacks with radius <= r)
        if 'attack_radius' in valid_results.columns:
            robust_samples = valid_results[
                (valid_results['correct'] == True) & 
                (valid_results['attack_radius'] > r)
            ]
        else:
            # Fallback: use certification radius
            robust_samples = valid_results[
                (valid_results['correct'] == True) & 
                (valid_results['radius'] > r)
            ]
        
        robust_accuracy = len(robust_samples) / total_samples if total_samples > 0 else 0
        robust_accuracies.append(robust_accuracy)
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.style.use('default')  # Use default style
    
    # Plot the curve
    plt.plot(radius_points, robust_accuracies, 'b-', linewidth=2.5)
    
    # Customize the plot
    plt.xlabel('Attack radius', fontsize=14)
    plt.ylabel('Empirical robust accuracy (g)', fontsize=14)
    plt.title('Empirical Robust Accuracy (g)', fontsize=16, fontweight='bold')
    
    # Set axis limits and grid
    plt.xlim(0, max_radius)
    plt.ylim(0, 1.0)
    plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Add some styling
    plt.tick_params(axis='both', which='major', labelsize=12)
    
    # Save the plot
    plot_path = os.path.join(output_dir, 'empirical_robust_accuracy.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.savefig(plot_path.replace('.png', '.pdf'), bbox_inches='tight')  # Also save as PDF
    
    print(f"Empirical robust accuracy plot saved to: {plot_path}")
    
    # Also create a detailed version with additional information
    plt.figure(figsize=(12, 8))
    
    # Main plot
    plt.subplot(2, 1, 1)
    plt.plot(radius_points, robust_accuracies, 'b-', linewidth=2.5, label='Robust Accuracy')
    plt.xlabel('Attack radius', fontsize=12)
    plt.ylabel('Empirical robust accuracy (g)', fontsize=12)
    plt.title('Empirical Robust Accuracy vs Attack Radius', fontsize=14, fontweight='bold')
    plt.xlim(0, max_radius)
    plt.ylim(0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Statistics subplot
    plt.subplot(2, 1, 2)
    if 'attack_radius' in valid_results.columns:
        # Histogram of attack radii
        attack_radii = valid_results['attack_radius'][valid_results['attack_radius'] != np.inf]
        plt.hist(attack_radii, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        plt.xlabel('Attack radius', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('Distribution of Successful Attack Radii', fontsize=12)
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    detailed_plot_path = os.path.join(output_dir, 'empirical_robust_accuracy_detailed.png')
    plt.savefig(detailed_plot_path, dpi=300, bbox_inches='tight')
    
    print(f"Detailed plot saved to: {detailed_plot_path}")
    
    # Print some statistics
    print(f"\n=== ROBUST ACCURACY STATISTICS ===")
    print(f"Total samples: {total_samples}")
    print(f"Samples with successful attacks: {len(valid_results[valid_results['best_audi'] != np.inf])}")
    print(f"Overall accuracy: {np.mean(valid_results['correct']):.4f}")
    print(f"Robust accuracy at radius 0: {robust_accuracies[0]:.4f}")
    print(f"Robust accuracy at radius {max_radius/2}: {robust_accuracies[len(radius_points)//2]:.4f}")
    print(f"Robust accuracy at radius {max_radius}: {robust_accuracies[-1]:.4f}")
    
    if len(attack_radii) > 0:
        print(f"Mean attack radius: {np.mean(attack_radii):.4f}")
        print(f"Median attack radius: {np.median(attack_radii):.4f}")
        print(f"Min attack radius: {np.min(attack_radii):.4f}")
        print(f"Max attack radius: {np.max(attack_radii):.4f}")
    
    plt.show()
    
    return radius_points, robust_accuracies


if __name__ == '__main__':
    from model_cn_celeb_2025.sb_interfaces import Classification, Verification
    from certification.audibility import Audibility
    from certification.vmoat import Vmoat

    root_path = '/mnt/data/old_home/strength/vmoat/vmoat_code25/vmoat_code2025'
    sigma = 40
    model_src = root_path + '/model/trained_models/CKPT+2023-07-11+09-25-42+00'
    hparams_src = 'test_ecapa_tdnn.yaml'
    savedir = root_path + '/model/trained_models/CKPT+2023-07-11+09-25-42+00'

    # ============================================================================
    # MULTI-CLASS SETUP: Use Classification instead of Verification
    # ============================================================================
    
    classification = Classification.from_hparams(
                                source=model_src,
                                hparams_file=hparams_src,
                                savedir=savedir,
                                run_opts = {
                                    "device": device,
                                    "data_parallel_backend": True,
                                })
    classification.eval()

    # Create or load VoxCeleb multi-class CSV
    multiclass_csv_path = './voxceleb_multiclass_dataset.csv'
    
    # Generate VoxCeleb dataset if it doesn't exist
    if not os.path.exists(multiclass_csv_path):
        print("Generating VoxCeleb multiclass dataset...")
        # Run the generation script
        import subprocess
        result = subprocess.run(['python', 'generate_voxceleb_csv.py'], capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error generating dataset: {result.stderr}")
            exit(1)
        print("VoxCeleb dataset generated successfully")
    
    # Load the VoxCeleb dataset
    multiclass_df = pd.read_csv(multiclass_csv_path)
    print(f"Loaded VoxCeleb dataset with {len(multiclass_df)} audio files")
    
    # No path replacement needed as VoxCeleb paths are already correct
    
    # Validate and clean the CSV data
    print("Validating audio file paths...")
    multiclass_df = validate_and_clean_csv(multiclass_df)
    
    # Get unique speakers for enrollment
    unique_speakers = multiclass_df['speaker_id'].unique()
    print(f"Found {len(unique_speakers)} unique speakers: {unique_speakers}")
    
    # Create enrollment list - use files marked as 'enroll'
    enroll_list_dict = []
    speaker_to_idx = {}  # Map speaker ID to index
    
    enroll_df = multiclass_df[multiclass_df['usage'] == 'enroll']
    
    for idx, speaker_id in enumerate(unique_speakers):
        speaker_to_idx[speaker_id] = idx
        # Get enrollment files for this speaker
        speaker_enroll_files = enroll_df[enroll_df['speaker_id'] == speaker_id]
        
        if len(speaker_enroll_files) > 0:
            # Use first enrollment file for this speaker
            enroll_file = speaker_enroll_files.iloc[0]['audio_path']
            enroll_list_dict.append({
                'wav_path': enroll_file,
                'spk_id': speaker_id
            })
    
    print(f"Created enrollment list with {len(enroll_list_dict)} speakers")
    
    # Enroll all speakers
    print("Enrolling speakers...")
    classification.enroll_many(enroll_list_dict)
    
    # Create smoothed version for certification
    num_classes = len(unique_speakers)
    smoothed_classification = Vmoat(classification, num_classes=num_classes, sigma=sigma)
    
    # Prepare test data - use files marked as 'trial'
    test_data = []
    trial_df = multiclass_df[multiclass_df['usage'] == 'trial']
    
    # 修复版本 - 不要过滤，测试所有样本
    print(f"Creating test samples from {len(trial_df)} trial samples...")

    test_samples = []

    for idx, (_, row) in enumerate(trial_df.iterrows()):
        speaker_id = row['speaker_id']
        audio_path = row['audio_path']

        # 确保说话人在speaker_to_idx中（应该都在）
        if speaker_id not in speaker_to_idx:
            print(f"Warning: Speaker {speaker_id} not in speaker_to_idx, adding it...")
            speaker_to_idx[speaker_id] = len(speaker_to_idx)
            unique_speakers.append(speaker_id)

        test_sample = {
            'audio_path': audio_path,
            'true_speaker_id': speaker_id,
            'true_speaker_idx': speaker_to_idx[speaker_id]
        }
        test_samples.append(test_sample)

        # 可选：限制样本数量（用于快速测试）
        if len(test_samples) >= 500:  # 先测试500个样本
            print(f"Limited to first {len(test_samples)} samples for testing")
            break

    print(f"Created {len(test_samples)} test samples")

    # 验证说话人分布
    test_speaker_distribution = {}
    for sample in test_samples:
        speaker = sample['true_speaker_id']
        test_speaker_distribution[speaker] = test_speaker_distribution.get(speaker, 0) + 1

    print(f"Test samples cover {len(test_speaker_distribution)} speakers")
    print(f"First 10 speakers in test: {list(test_speaker_distribution.keys())[:10]}")
    
    # Results storage
    results = {
        'audio_path': [],
        'true_speaker_id': [],
        'predicted_speaker_idx': [],
        'predicted_speaker_id': [],
        'radius': [],
        'correct': [],
        'best_audi': []
    }
    # Main testing loop
    for idx in tqdm(range(min(len(test_data), 60))):  # Test up to 60 samples (15 speakers x 4 test files each)
        test_sample = test_data[idx]
        
        print(f'Processing: {test_sample["audio_path"]}')
        print(f'True speaker: {test_sample["true_speaker_id"]} (idx: {test_sample["true_speaker_idx"]})')
        
        # Check if file exists
        if not os.path.exists(test_sample['audio_path']):
            print(f"File not found: {test_sample['audio_path']}, skipping...")
            continue
        
        # Load test audio with error handling
        try:
            wav_x_trial, _ = torchaudio.load(test_sample['audio_path'])
        except Exception as e:
            print(f"Error loading audio {test_sample['audio_path']}: {e}, skipping...")
            continue
        
        # Ensure minimum length
        # if wav_x_trial.size(1) < 24*2048:
        if wav_x_trial.size(1) < 62.5*2048:   # 8s

            print(f"Audio too short: {wav_x_trial.size(1)}, skipping...")
            continue
            
        # wav_x_trial = wav_x_trial[:, -24*2048:]  # Use last 24*2048 samples
        wav_x_trial = wav_x_trial[:, -62.5*2048:]  # Use last 62.5*2048 samples 8s

        # Perform adversarial attack
        target_speaker_idx = test_sample['true_speaker_idx']
        
        adv_wav_x, best_audi = pgd_attack_psy_smooth_multiclass(
                            smoothed_classification, 
                            wav_x_trial.to(device), 
                            target_speaker_idx,         
                            eps=0.1, 
                            alpha=0.01, 
                            iters=100,  # Reduced for faster testing
                            device=device
                        )
        
        # Get prediction and certification
        prediction, radius = smoothed_classification.certify(wav_x_trial.to(device), 
                                            args.N0, args.N, args.alpha, args.batch)
        
        # Convert prediction index back to speaker ID
        if prediction != -1 and prediction < len(unique_speakers):
            predicted_speaker_id = unique_speakers[prediction]
        else:
            predicted_speaker_id = "ABSTAIN"
        
        # Store results
        results['audio_path'].append(test_sample['audio_path'])
        results['true_speaker_id'].append(test_sample['true_speaker_id'])
        results['predicted_speaker_idx'].append(prediction)
        results['predicted_speaker_id'].append(predicted_speaker_id)
        results['radius'].append(radius)
        results['correct'].append(prediction == test_sample['true_speaker_idx'])
        results['best_audi'].append(best_audi)
        
        print(f'True: {test_sample["true_speaker_id"]}, Predicted: {predicted_speaker_id}, Radius: {radius:.4f}')
        print(f'Attack audibility: {best_audi:.4f}')
        print('-' * 50)
    
    # Save results
    output_dir = 'results_multiclass_classification'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    csv_name = os.path.join(output_dir, f'multiclass_results_sigma_{sigma}_N_{args.N}_alpha_{args.alpha}.csv')
    results_df = pd.DataFrame(results)
    results_df.to_csv(csv_name, index=False)
    
    # Print summary
    total_samples = len(results['correct'])
    correct_predictions = sum(results['correct'])
    accuracy = correct_predictions / total_samples if total_samples > 0 else 0
    
    print(f"\n=== MULTI-CLASS CLASSIFICATION RESULTS ===")
    print(f"Total samples: {total_samples}")
    print(f"Correct predictions: {correct_predictions}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Results saved to: {csv_name}")
    
    # Generate Empirical Robust Accuracy plot
    print(f"\n=== GENERATING ROBUST ACCURACY PLOT ===")
    if total_samples > 0:
        radius_points, robust_accuracies = plot_empirical_robust_accuracy(
            results_df, 
            output_dir, 
            max_radius=5.0, 
            num_points=50
        )
        
        # Save the plot data as well
        plot_data = pd.DataFrame({
            'radius': radius_points,
            'robust_accuracy': robust_accuracies
        })
        plot_data_path = os.path.join(output_dir, 'robust_accuracy_data.csv')
        plot_data.to_csv(plot_data_path, index=False)
        print(f"Plot data saved to: {plot_data_path}")
    else:
        print("No samples to plot.")

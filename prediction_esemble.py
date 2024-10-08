import torch
import pandas as pd

pre_df = pd.read_csv("/scratch/zguo32/cxr-lt-2024/test/test_task1_sample_submission.csv")
submit_df = pd.read_csv("/scratch/zguo32/cxr-lt-2024/test/test_task1.csv") #df convnext-small-224
#pred_convnext_small_384 = torch.load('pred.pt')
#df_convnext_small_384 = pd.DataFrame(pred_convnext_small_384, columns=submit_df.columns[-41:-1])

tmp = torch.load("predictions_bc224.pt")
single_base_clip_224_df = pd.DataFrame(tmp, columns=pre_df.columns[-41:-1])

fusion_tiny_224_df = pd.read_csv("/scratch/zguo32/cxr-lt-2024/test/tmp1_test_task1_submission_fusion_t224.csv") #df
fusion_tiny_384_df = pd.read_csv("/scratch/zguo32/cxr-lt-2024/test/tmp1_test_task1_submission_fusion_t384.csv")
fusion_small_224_df = pd.read_csv("/scratch/zguo32/cxr-lt-2024/test/tmp1_test_task1_submission_fusion_s224.csv") #df
fusion_small_384_df = pd.read_csv("/scratch/zguo32/cxr-lt-2024/test/tmp1_test_task1_submission_fusion_s384.csv")
#fusion_base_224_df = pd.read_csv("/scratch/zguo32/cxr-lt-2024/test/tmp1_test_task1_submission_fusion_b224.csv")
fusion_base_384_df = pd.read_csv("/scratch/zguo32/cxr-lt-2024/test/tmp1_test_task1_submission_fusion_b384.csv")
fusion_base_clip_224 = pd.read_csv("/scratch/zguo32/cxr-lt-2024/test/tmp1_test_task1_submission_fusion_bc224.csv")
fusion_base_clip_384 = pd.read_csv("/scratch/zguo32/cxr-lt-2024/test/tmp1_test_task1_submission_fusion_bc384.csv")
fusion_v2_base_384_df = pd.read_csv("/scratch/zguo32/cxr-lt-2024/test/tmp1_test_task1_submission_fusion_v2b384.csv")
#fusion_base_224_v2_df = pd.read_csv("/scratch/zguo32/cxr-lt-2024/test/tmp1_test_task1_submission_fusion_v2b224.csv")

if 0:
    single_base_clip_224_df.iloc[:,1:41] 

if 0:
    best_esemble = ( fusion_tiny_224_df.iloc[:,1:41] + fusion_tiny_384_df.iloc[:,1:41] + fusion_small_224_df.iloc[:,1:41] + \
                    fusion_small_384_df.iloc[:,1:41] + fusion_v2_base_384_df.iloc[:,1:41] + fusion_base_clip_384.iloc[:,1:41] + \
                    fusion_base_384_df.iloc[:,1:41] )/ 7 #+ fusion_base_224_v2_df.iloc[:,1:41]) / 8
    best_esemble.insert(0,'dicom_id', submit_df['dicom_id'])
    best_esemble.to_csv("/scratch/zguo32/cxr-lt-2024/test/save/tmp1_test_task1_submission_fusion_t224_t384_s224_s384_b384_bc384_v2b384.csv", \
                         index=False) 

if 1:
    best_esemble = ( fusion_tiny_224_df.iloc[:,1:41] + fusion_tiny_384_df.iloc[:,1:41] + fusion_small_224_df.iloc[:,1:41] + \
                    fusion_small_384_df.iloc[:,1:41] + fusion_v2_base_384_df.iloc[:,1:41] + fusion_base_clip_384.iloc[:,1:41] + \
                    fusion_base_384_df.iloc[:,1:41] ) / 7
    best_esemble.insert(0,'dicom_id', submit_df['dicom_id'])
    best_esemble.to_csv("/scratch/zguo32/cxr-lt-2024/test/save/tmp1_test_task1_submission_fusion_t224_t384_s224_s384_b384_bc384_v2b384.csv", \
                         index=False) 



if 0:
    best_esemble = ( fusion_small_224_df.iloc[:,1:41] + fusion_small_384_df.iloc[:,1:41] + fusion_v2_base_384_df.iloc[:,1:41] + \
                    fusion_base_clip_384.iloc[:,1:41] + fusion_base_384_df.iloc[:,1:41] ) / 5
    best_esemble.insert(0,'dicom_id', submit_df['dicom_id'])
    best_esemble.to_csv("/scratch/zguo32/cxr-lt-2024/development_task1_submission_fusion_s224_s384_v2b384_bc384_b384_tmp45.csv", \
                         index=False) 


if 0:
    single_small_224_df = pd.read_csv('/scratch/zguo32/cxr-lt-2024/development_task1_submission_224_aug+_ga_v2_tmp32.csv')
    single_base_clip_384_df = pd.read_csv('/scratch/zguo32/cxr-lt-2024/development_task1_submission_base_clip_384_aug+_ga_v2_tmp34.csv')
    tmp = torch.load("predictions_convnextv2_base_384_aug+_ga.pt")
    single_v2b384_df = pd.DataFrame(tmp, columns=submit_df.columns[-41:-1])

    six_mix_esemble = ( fusion_small_224_df.iloc[:,1:41] + fusion_small_384_df.iloc[:,1:41] + fusion_v2_base_384_df.iloc[:,1:41] + \
                    fusion_base_clip_384.iloc[:,1:41] + fusion_base_384_df.iloc[:,1:41] + single_base_clip_384_df.iloc[:,1:41] ) / 6
    six_mix_esemble.insert(0,'dicom_id', submit_df['dicom_id'])
    six_mix_esemble.to_csv("/scratch/zguo32/cxr-lt-2024/development_task1_submission_fusion_s224_s384_v2b384_bc384_b384_single_bc384_tmp47.csv", \
                         index=False)

if 0:
    seven_mix_esemble = ( fusion_small_224_df.iloc[:,1:41] + fusion_small_384_df.iloc[:,1:41] + fusion_v2_base_384_df.iloc[:,1:41] + \
                    fusion_base_clip_384.iloc[:,1:41] + fusion_base_384_df.iloc[:,1:41] + single_base_clip_384_df.iloc[:,1:41] + \
                    single_small_224_df.iloc[:,1:41] ) / 7
    seven_mix_esemble.insert(0,'dicom_id', submit_df['dicom_id'])
    seven_mix_esemble.to_csv("/scratch/zguo32/cxr-lt-2024/development_task1_submission_fusion_s224_s384_v2b384_bc384_b384_single_bc384_s224_tmp48.csv", \
                         index=False) 


if 0:
    eight_mix_esemble = ( fusion_small_224_df.iloc[:,1:41] + fusion_small_384_df.iloc[:,1:41] + fusion_v2_base_384_df.iloc[:,1:41] + \
                    fusion_base_clip_384.iloc[:,1:41] + fusion_base_384_df.iloc[:,1:41] + single_base_clip_384_df.iloc[:,1:41] + \
                    single_small_224_df.iloc[:,1:41] + single_v2b384_df ) / 8
    eight_mix_esemble.insert(0,'dicom_id', submit_df['dicom_id'])
    eight_mix_esemble.to_csv("/scratch/zguo32/cxr-lt-2024/development_task1_submission_fusion_s224_s384_v2b384_bc384_b384_single_bc384_s224_v2b384_tmp49.csv", \
                         index=False)  



#five_mix_emsemble = ( fusion_small_224_df.iloc[:,1:41] + fusion_small_384_df.iloc[:,1:41] + fusion_v2_base_384_df.iloc[:,1:41] + single_small_224_df.iloc[:,1:41] + single_base_clip_384_df.iloc[:,1:41]) / 5
#five_mix_emsemble.insert(0,'dicom_id', submit_df['dicom_id'])
#five_mix_emsemble.to_csv("/scratch/zguo32/cxr-lt-2024/development_task1_submission_fusion_s224_s384_v2b384_single_s224_bc_384_tmp39.csv", index=False) 
#five_esemble = df_convnext_small_384
#five_esemble.insert(0,'dicom_id', submit_df['dicom_id'])
#five_esemble.to_csv("/scratch/zguo32/cxr-lt-2024/development_task1_submission_", index=False) 
import torch
import pandas as pd

pre_df = pd.read_csv("...path_to_your_file/test_task1_sample_submission.csv")
submit_df = pd.read_csv("...path_to_your_file/test_task1.csv") 

fusion_tiny_224_df = pd.read_csv("...path_to_your_file/tmp1_test_task1_submission_fusion_t224.csv")
fusion_tiny_384_df = pd.read_csv("...path_to_your_file/tmp1_test_task1_submission_fusion_t384.csv")
fusion_small_224_df = pd.read_csv("...path_to_your_file/tmp1_test_task1_submission_fusion_s224.csv")
fusion_small_384_df = pd.read_csv("...path_to_your_file/tmp1_test_task1_submission_fusion_s384.csv")
#fusion_base_224_df = pd.read_csv("...path_to_your_file/tmp1_test_task1_submission_fusion_b224.csv")
fusion_base_384_df = pd.read_csv("...path_to_your_file/tmp1_test_task1_submission_fusion_b384.csv")
fusion_base_clip_224 = pd.read_csv("...path_to_your_file/tmp1_test_task1_submission_fusion_bc224.csv")
fusion_base_clip_384 = pd.read_csv("...path_to_your_file/tmp1_test_task1_submission_fusion_bc384.csv")
fusion_v2_base_384_df = pd.read_csv("...path_to_your_file/tmp1_test_task1_submission_fusion_v2b384.csv")

best_esemble = ( fusion_tiny_224_df.iloc[:,1:41] + fusion_tiny_384_df.iloc[:,1:41] + fusion_small_224_df.iloc[:,1:41] + \
                    fusion_small_384_df.iloc[:,1:41] + fusion_v2_base_384_df.iloc[:,1:41] + fusion_base_clip_224.iloc[:,1:41] \
                    fusion_base_clip_384.iloc[:,1:41] + fusion_base_384_df.iloc[:,1:41] ) / 8
best_esemble.insert(0,'dicom_id', submit_df['dicom_id'])
best_esemble.to_csv("...path_to_your_file/tmp1_test_task1_submission_fusion_t224_t384_s224_s384_b384_bc224_bc384_v2b384.csv", \
                         index=False) 







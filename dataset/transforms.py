import cv2
import albumentations as A

def get_transforms(size):
    transforms_train = A.Compose([
        A.RandomResizedCrop(size,size, scale=(0.9, 1), p=1, interpolation=cv2.INTER_LANCZOS4), 
        A.HorizontalFlip(p=0.5),
        #A.ShiftScaleRotate(p=0.5),
        #A.RandomBrightnessContrast(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.85),
        A.RandomBrightnessContrast(p=0.75),
        A.OneOf([
            A.OpticalDistortion(),
            A.GridDistortion(),
            A.ElasticTransform(),
        ], p=0.2),
        #albumentations.OneOf([
        #    albumentations.MotionBlur(blur_limit=5),
        #    albumentations.MedianBlur(blur_limit=5),
        #    albumentations.GaussianBlur(blur_limit=5),
        #    albumentations.GaussNoise(var_limit=(5.0, 30.0)),
        #], p=0.7),
        #albumentations.OneOf([
        #    albumentations.OpticalDistortion(distort_limit=1.0),
        #    albumentations.GridDistortion(num_steps=5, distort_limit=1.),
        #    albumentations.ElasticTransform(alpha=3),
        #], p=0.7),
        A.OneOf([
            A.GaussNoise(),
            A.GaussianBlur(),
            A.MotionBlur(),
            A.MedianBlur(),
        ], p=0.2),
        A.CLAHE(clip_limit=4.0, p=0.7), #1 from https://github.com/haqishen/SIIM-ISIC-Melanoma-Classification-1st-Place-Solution/blob/master/dataset.py
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5), #2
        #A.Cutout(max_h_size=int(size * 0.375), max_w_size=int(size * 0.375), num_holes=1, p=0.7), #3 depercated
        A.CoarseDropout(max_holes=10, max_height=int(size * 0.375), max_width=int(size * 0.375), p=0.75), #4S
        A.Resize(size,size, interpolation=cv2.INTER_LANCZOS4),
        A.Normalize(),
    ])

    transforms_val = A.Compose([
        A.Resize(size,size, interpolation=cv2.INTER_LANCZOS4),
        A.Normalize()
    ])
    return transforms_train, transforms_val


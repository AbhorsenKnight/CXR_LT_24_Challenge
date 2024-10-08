import torch
import timm
import torch.nn as nn
import copy    
import einops
from model.ml_decoder import MLDecoder
from positional_encodings.torch_encodings import PositionalEncoding2D, Summer


class Backbone(nn.Module):
    def __init__(self, timm_init_args):
        super().__init__()
        if "convnext" in timm_init_args['model_name']:
            #if "convnext_base" or "convnextv2_base" in timm_init_args['model_name']:
            if "base" in timm_init_args['model_name']:
                self.model = timm.create_model(**timm_init_args, global_pool='')
                self.pos_encoding = Summer(PositionalEncoding2D(1024))
                self.head = MLDecoder(num_classes=40, initial_num_features=1024)
            else:
                #self.model = timm.create_model(**timm_init_args, global_pool='')
                #self.pos_encoding = Summer(PositionalEncoding2D(768))
                self.model = timm.create_model(**timm_init_args)
                self.model.head = nn.Identity()
                self.pos_encoding = Summer(PositionalEncoding2D(768))
                self.head = MLDecoder(num_classes=40, initial_num_features=768)
        elif "efficientnet" in timm_init_args['model_name']:
            self.model = timm.create_model(**timm_init_args, global_pool='')
            self.pos_encoding = Summer(PositionalEncoding2D(1536))
            self.head = MLDecoder(num_classes=40, initial_num_features=1536)
        elif "resnet50" in timm_init_args['model_name']:
            self.model = timm.create_model(**timm_init_args, global_pool='')
            #self.model.head = nn.Identity()
            self.pos_encoding = Summer(PositionalEncoding2D(2048))
            self.head = MLDecoder(num_classes=40, initial_num_features=2048)
        elif "resnext" in timm_init_args['model_name']:
            self.model = timm.create_model(**timm_init_args, global_pool='')
            self.pos_encoding = Summer(PositionalEncoding2D(2048))
            self.head = MLDecoder(num_classes=40, initial_num_features=2048)
        elif "densenet" in timm_init_args['model_name']:
            self.model = timm.create_model(**timm_init_args, global_pool='')
            self.pos_encoding = Summer(PositionalEncoding2D(1024))
            self.head = MLDecoder(num_classes=40, initial_num_features=1024)

    def forward(self, x):
        x = self.model(x)
        x = self.pos_encoding(x)
        x = self.head(x)
        return x


class FusionBackbone(nn.Module):
    def __init__(self, timm_init_args, pretrained_path=None):
        super().__init__()

        print(pretrained_path)

        num_of_features = 1024
        if "convnext" in timm_init_args['model_name']:
            if "base" in timm_init_args['model_name']:
                num_of_features = 1024
            else:
                num_of_features = 768                
        elif "efficientnet" in timm_init_args['model_name']:
            num_of_features = 1536
        elif "resnet50" in timm_init_args['model_name'] or "resnext" in timm_init_args['model_name']:
            num_of_features = 2048

        self.model = timm.create_model(**timm_init_args)
        #self.model = timm.create_model(**timm_init_args, global_pool='')
        self.model.head = MLDecoder(num_classes=40, initial_num_features=num_of_features)
        if pretrained_path is not None:
            print("loading pretrained weight")
            self.model.load_state_dict(torch.load(pretrained_path), strict=False)
        self.model.head = nn.Identity()
        self.conv2d = nn.Conv2d(num_of_features, num_of_features, kernel_size=3, stride=2, padding=1)
        self.pos_encoding = Summer(PositionalEncoding2D(num_of_features))
        self.padding_token = nn.Parameter(torch.randn(1, num_of_features, 1, 1))
        self.segment_embedding = nn.Parameter(torch.randn(4, num_of_features, 1, 1))
                
        self.head = MLDecoder(num_classes=40, initial_num_features=num_of_features)
        self.transformer_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=num_of_features, nhead=8), num_layers=2)



        '''
        if "convnext" in timm_init_args['model_name']:
            if "base" in timm_init_args['model_name']:
                self.model = timm.create_model(**timm_init_args, global_pool='')
                self.model.head = MLDecoder(num_classes=40, initial_num_features=1024)
                if pretrained_path is not None:
                    print("loading pretrained weight")
                    self.model.load_state_dict(torch.load(pretrained_path), strict=False)
                self.model.head = nn.Identity()
                self.conv2d = nn.Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1)
                self.pos_encoding = Summer(PositionalEncoding2D(1024))
                self.padding_token = nn.Parameter(torch.randn(1, 1024, 1, 1))
                self.segment_embedding = nn.Parameter(torch.randn(4, 1024, 1, 1))
                
                self.head = MLDecoder(num_classes=40, initial_num_features=1024)
                self.transformer_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=1024, nhead=8), num_layers=2)
            else:
                self.model = timm.create_model(**timm_init_args, global_pool='')
                self.model.head = MLDecoder(num_classes=40, initial_num_features=768)
                if pretrained_path is not None:
                    print("loading pretrained weight")
                    self.model.load_state_dict(torch.load(pretrained_path), strict=False)
                self.model.head = nn.Identity()
                self.conv2d = nn.Conv2d(768, 768, kernel_size=3, stride=2, padding=1)
                self.pos_encoding = Summer(PositionalEncoding2D(768))
                self.padding_token = nn.Parameter(torch.randn(1, 768, 1, 1))
                self.segment_embedding = nn.Parameter(torch.randn(4, 768, 1, 1))
                    
                self.head = MLDecoder(num_classes=40, initial_num_features=768)
                self.transformer_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=768, nhead=8), num_layers=2)
        elif "efficientnet" in timm_init_args['model_name']:
            self.model = timm.create_model(**timm_init_args, global_pool='')
            #self.model.head = nn.Identity()
            self.pos_encoding = Summer(PositionalEncoding2D(1536))
            self.head = MLDecoder(num_classes=40, initial_num_features=1536)
        elif "resnet50" in timm_init_args['model_name']:
            self.model = timm.create_model(**timm_init_args, global_pool='')
            #self.model.head = nn.Identity()
            self.pos_encoding = Summer(PositionalEncoding2D(2048))
            self.head = MLDecoder(num_classes=40, initial_num_features=2048)


        #self.model = timm.create_model(**timm_init_args)
        #self.model.head = MLDecoder(num_classes=40, initial_num_features=768)
        '''

    def forward(self, x):
        
        #print(x.shape)

        b, s, _, _, _ = x.shape

        x = einops.rearrange(x, 'b s c h w -> (b s) c h w')
        no_pad = torch.nonzero(x.sum(dim=(1, 2, 3)) != 0).squeeze(1)
        x = x[no_pad]

        with torch.no_grad():
            x = self.model(x).detach()
        
        #x = self.model(x)

        x = self.conv2d(x)
        x = self.pos_encoding(x)    

        pad_tokens = einops.repeat(self.padding_token, '1 c 1 1 -> (b s) c h w', b=b, s=s, h=x.shape[2], w=x.shape[3]).type_as(x)
        segment_embedding = einops.repeat(self.segment_embedding, 's c 1 1 -> (b s) c h w', b=b, h=x.shape[2], w=x.shape[3]).type_as(x)
        pad_tokens[no_pad] = x + segment_embedding[no_pad]
        x = pad_tokens

        x = einops.rearrange(x, '(b s) c h w -> b (s h w) c', b=b, s=s, h=x.shape[2], w=x.shape[3])
        # mask =(x.sum(dim=-1) == 0).transpose(0, 1)
        mask =(x.sum(dim=-1) == 0).transpose(0, 1)
        x = self.transformer_encoder(x, src_key_padding_mask=mask)

        #print('shape of mask')
        #print(mask.shape)

        #x = self.head(x, mask) try to fix the size mismatch issue
        x = self.head(x, mask.transpose(0, 1))

        return x

class FusionBackboneTask2(nn.Module):
    def __init__(self, timm_init_args, pretrained_path=None):
        super().__init__()

        print(pretrained_path)
        num_of_features = 1024
        if "convnext" in timm_init_args['model_name']:
            if "base" in timm_init_args['model_name']:
                num_of_features = 1024
            else:
                num_of_features = 768                
        elif "efficientnet" in timm_init_args['model_name']:
            num_of_features = 1536
        elif "resnet50" in timm_init_args['model_name']:
            num_of_features = 2048

        self.model = timm.create_model(**timm_init_args, global_pool='')
        self.model.head = nn.Identity()
        self.conv2d = nn.Conv2d(num_of_features, num_of_features, kernel_size=3, stride=2, padding=1)
        self.pos_encoding = Summer(PositionalEncoding2D(num_of_features))
        self.padding_token = nn.Parameter(torch.randn(1, num_of_features, 1, 1))
        self.segment_embedding = nn.Parameter(torch.randn(4, num_of_features, 1, 1))                
        self.head = MLDecoder(num_classes=40, initial_num_features=num_of_features)
        self.transformer_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=num_of_features, nhead=8), num_layers=2)
        if pretrained_path is not None:
            print("loading pretrained weight")
            self.model.load_state_dict(torch.load(pretrained_path), strict=False)
        self.head = MLDecoder(num_classes=26, initial_num_features=num_of_features)
        self.transformer_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=num_of_features, nhead=8), num_layers=2)


        #self.model = timm.create_model(**timm_init_args)
        #self.model.head = MLDecoder(num_classes=40, initial_num_features=768)


    def forward(self, x):
        
        #print(x.shape)

        b, s, _, _, _ = x.shape

        x = einops.rearrange(x, 'b s c h w -> (b s) c h w')
        no_pad = torch.nonzero(x.sum(dim=(1, 2, 3)) != 0).squeeze(1)
        x = x[no_pad]

        with torch.no_grad():
            x = self.model(x).detach()
        
        #x = self.model(x)

        x = self.conv2d(x)
        x = self.pos_encoding(x)    

        pad_tokens = einops.repeat(self.padding_token, '1 c 1 1 -> (b s) c h w', b=b, s=s, h=x.shape[2], w=x.shape[3]).type_as(x)
        segment_embedding = einops.repeat(self.segment_embedding, 's c 1 1 -> (b s) c h w', b=b, h=x.shape[2], w=x.shape[3]).type_as(x)
        pad_tokens[no_pad] = x + segment_embedding[no_pad]
        x = pad_tokens

        x = einops.rearrange(x, '(b s) c h w -> b (s h w) c', b=b, s=s, h=x.shape[2], w=x.shape[3])
        # mask =(x.sum(dim=-1) == 0).transpose(0, 1)
        mask =(x.sum(dim=-1) == 0).transpose(0, 1)
        x = self.transformer_encoder(x, src_key_padding_mask=mask)

        #print('shape of mask')
        #print(mask.shape)

        #x = self.head(x, mask) try to fix the size mismatch issue
        x = self.head(x, mask.transpose(0, 1))

        return x


class PretrainedBackbone(nn.Module):
    def __init__(self, timm_init_args, pretrained_path):
        super().__init__()
        self.model = timm.create_model(**timm_init_args)
        self.new_head = copy.deepcopy(self.model.head)
        self.model.load_state_dict(torch.load(pretrained_path))
        self.model.head = nn.Identity()

    def forward(self, x):
        with torch.no_grad():
            x = self.model(x)
        x = self.new_head(x.detach())
        return x

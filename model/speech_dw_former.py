
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce
from module.utils import create_PositionalEncoding, statistical_information
from module.utils import _no_grad_trunc_normal_
from module.utils import normalize_tensor
from module.speech_dw_former_layer import SpeechDW_UnitEncoder
from model.transformer import build_vanilla_transformer
import math
from loguru import logger
from module.classifier_layer import SVM_Classifier

class ModifiedDWBlock(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        
        self.num_heads = kwargs["num_heads"]
        self.vanilla_transformer = build_vanilla_transformer(official=False, **kwargs)
        self.DLWT = build_vanilla_transformer(official=False, **kwargs)
    
    def forward(self, x):
        output_vanilla, attn_weights_vanilla = self.vanilla_transformer(x = x, 
                                                                       need_weights = True,
                                                                       need_classifier = False)
        # attn_weights_vanilla: (bsz, tgt_len, src_len) -> (bsz, src_len)
        attn_weights_vanilla = attn_weights_vanilla[0].sum(dim=1) # sum along row
        # logger.info(f"Heads average + Sum along row:\n{attn_weights_vanilla}")
        # attn_weights_vanilla = F.softmax(attn_weights_vanilla, dim=-1)
        attn_weights_vanilla = normalize_tensor(attn_weights_vanilla)
        # attn_weights_vanilla[0][5] = 0.6
        # attn_weights_vanilla[1][4] = 0.6
        # attn_weights_vanilla[1][5] = 0.6
        attn_mask, window_mapping = self.generate_local_window_mask_v2(attn_weights_vanilla, threshold=0.5)
        ### attn_mask: (bsz, tsz, tsz)
        ### window_mapping: (bsz, max_num_windows, tsz)

        attn_mask = attn_mask.unsqueeze(1)
        attn_mask = attn_mask.expand(-1, self.num_heads, -1, -1)
        b, h, t1, t2 = attn_mask.size()
        new_shape = (b * h, t1, t2)
        attn_mask = attn_mask.reshape(*new_shape)
        ### attn_mask: (bsz * num_heads, tsz, tsz)
        ### 2 tokens i,j can see each other then value at ij = 0
        
        output_DLWT, attn_weights_local = self.DLWT(x = output_vanilla, 
                                                   need_weights = True, 
                                                   attn_mask = attn_mask,
                                                   need_classifier = False)
        ### output_DLWT: (bsz, tsz, fea_dim)
        ### output_DLWT: no feature inf
        attn_weights_local = torch.where(torch.isneginf(attn_weights_local[0]),
                                         torch.zeros_like(attn_weights_local[0]),
                                         attn_weights_local[0])
        
        # attn_weights_local: (bsz, tgt_len, src_len) -> (bsz, src_len)
        attn_weights_local = attn_weights_local.sum(dim=1) # sum along row
        ### why attn_weights_local is nan at 1st, 2nd, 4th in batch
        attn_weights_local = F.softmax(attn_weights_local, dim=-1)
        ### why attn_weights_local is nan?

        word_tokens = self.window_weighted_sum(output_DLWT, attn_weights_local, 
                                               window_mapping)
        # logger.info(f"Attention weights vanilla: {attn_weights_vanilla}")
        # logger.info(f"Attention weights local: {attn_weights_vanilla}")
        # logger.info(f"Window mapping: {window_mapping}")
        # logger.info(f"Input: {x}")

        return (torch.concat((word_tokens, x), dim=1), window_mapping)
        
    def generate_local_window_mask(self, weights, threshold=0.5):
        '''
        Inputs:
        - weights: (bsz, tsz)
        
        Returns:
        - 
        '''
        bsz, tsz = weights.shape
        attn_mask = torch.full((bsz, tsz, tsz), -float('inf'), device='cuda')
        window_mapping = torch.zeros((bsz, tsz, tsz), dtype=torch.float, device='cuda')
        
        max_window_count = 0
        for batch_cnt in range(bsz): 
            i = 0
            window_count = 0
            while i < tsz:
                if weights[batch_cnt, i] < threshold:
                    window_size = 0
                    while i + window_size < tsz and weights[batch_cnt, i + window_size] < threshold:
                        window_mapping[batch_cnt, window_count, i + window_size] = 1.0
                        window_size += 1
                    for j in range(i, i + window_size):
                        for k in range(i, i + window_size):
                            attn_mask[batch_cnt, j, k] = 0
                    i += window_size
                else:
                    window_size = 0
                    while i + window_size < tsz and weights[batch_cnt, i + window_size] >= threshold:
                        window_mapping[batch_cnt, window_count, i + window_size] = 1
                        window_size += 1
                    for j in range(i, i + window_size):
                        for k in range(i, i + window_size):
                            attn_mask[batch_cnt, j, k] = 0
                    i += window_size
                window_count += 1
            if max_window_count < window_count:
                max_window_count = window_count

        return attn_mask, window_mapping[:, :max_window_count, :]
    
    def generate_local_window_mask_v2(self, weights, threshold=0.5):
        '''
        Inputs:
        - weights: (bsz, tsz)
        
        Returns:
        - 
        '''
        bsz, num_frames = weights.shape
        attn_mask = torch.full((bsz, num_frames, num_frames), -float('inf'), device="cuda")
        window_mapping = torch.zeros((bsz, num_frames, num_frames), dtype=torch.float, device="cuda")
        weights = torch.ge(weights, threshold)

        max_window_count = 0
        for batch_cnt in range(bsz): 
            frame_start = 0
            window_count = 0
            for frame_cnt in range(1, num_frames):
                 # same window
                if not (weights[batch_cnt, frame_cnt] ^ weights[batch_cnt, frame_start]):
                    continue
                # current frame is begin of new window => check the window size = frame_cnt - frame_start
                if frame_start + 1 == frame_cnt:
                    weights[batch_cnt, frame_start] = weights[batch_cnt, frame_cnt]
                    continue
                
                attn_mask[batch_cnt, frame_start : frame_cnt, frame_start : frame_cnt] = 0
                window_mapping[batch_cnt, window_count, frame_start: frame_cnt] = 1
                
                frame_start = frame_cnt
                window_count += 1
            
            frame_cnt += 1
            attn_mask[batch_cnt, frame_start : frame_cnt, frame_start : frame_cnt] = 0
            window_mapping[batch_cnt, window_count, frame_start: frame_cnt] = 1
            
            frame_start = frame_cnt
            window_count += 1
            if max_window_count < window_count: 
                max_window_count = window_count

        return attn_mask, window_mapping[:, :max_window_count, :]
    
    def window_weighted_sum(self, output_DLWT, attn_weights_local, window_mapping):
        weighted_output = output_DLWT * attn_weights_local.unsqueeze(-1)
        window_weighted_sum = torch.matmul(window_mapping, weighted_output)
        return window_weighted_sum

class SpeechDW_EncoderBlock(nn.Module):
    def __init__(self, Encoder_args, num_layers=1,
                 #  num_blocks, embed_dim, ffn_embed_dim=2304, local_size=0, num_heads=8, 
                # dropout=0.1, attention_dropout=0.1, activation='relu', 
                use_position=False
                ):
        super().__init__()
        self.position = create_PositionalEncoding(Encoder_args["embed_dim"]) if use_position else None
        self.input_norm = nn.LayerNorm(Encoder_args["embed_dim"])
        self.layers = nn.ModuleList(
            # [SpeechDW_Encoder(embed_dim, ffn_embed_dim, local_size, num_heads, dropout, 
            #     attention_dropout, activation) for _ in range(num_layers)]
            [SpeechDW_UnitEncoder(
                embed_dim=Encoder_args["embed_dim"], 
                ffn_embed_dim=Encoder_args["ffn_embed_dim"], 
                local_size=Encoder_args["local_size"],
                num_heads=Encoder_args["num_heads"], 
                dropout=Encoder_args["dropout"], 
                attention_dropout=Encoder_args["attention_dropout"], 
                activation=Encoder_args["activation"]) 
             for _ in range(num_layers)
             ]
            )

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, inputs: tuple):
        x, window_mapping = inputs
        output = self.input_norm(x)

        for layer in self.layers:
            output = layer(output, window_mapping=window_mapping, x_position=self.position)
            # output = layer(output)

        return (output, window_mapping.shape[1])

class SpeechDW_MergeBlock(nn.Module):
    ''' Merge features between tow phases.

        The number of tokens is decreased while the dimension of token is increased.
    '''
    def __init__(self, in_channels, merge_scale, 
                #  num_wtok, 
                 expand=2):
        super().__init__()

        out_channels = in_channels * expand
        self.MS = merge_scale
        # self.num_wtok = num_wtok
        self.pool = nn.AdaptiveAvgPool2d((1, in_channels))
        self.fc = nn.Linear(in_channels, out_channels)
        self.norm = nn.LayerNorm(out_channels)
        
    def forward(self, inputs: tuple):
        x, num_wtok = inputs
        x_wtok, x_fea = x[:, :num_wtok], x[:, num_wtok:]

        B, T, C = x_fea.shape
        ms = T if self.MS == -1 else self.MS

        need_pad = T % ms
        if need_pad:
            pad = ms - need_pad
            x_fea = F.pad(x_fea, (0, 0, 0, pad), mode='constant', value=0)
            T += pad

        x_fea = x_fea.view(B, T//ms, ms, C)
        x_fea = self.pool(x_fea).squeeze(dim=-2)

        # x = torch.cat((x_wtok, x_fea), dim=1)
        x = x_fea
        x = self.norm(self.fc(x))

        return x

class SpeechDW_Former(nn.Module):
    def __init__(self, input_dim, ffn_embed_dim, num_heads, num_classes,
                 num_layers: list, num_layers_modified_dw: list,
                 hop, expand, dropout=0.1, attention_dropout=0.1, 
                 classifier=None,
                 **kwargs):
        '''
        input_dim: 
        num_layers: []
        Locals: []
        Merge: []
        '''
        super().__init__()
        
        self.input_dim = input_dim // num_heads * num_heads
        Locals, Merge = statistical_information(hop)
        assert isinstance(num_layers, list)
        
        ### 31/03 fix start: remove initialized wtok
        # self.num_wtok = math.ceil(kwargs['length'] / Merge[-2])

        # self.wtok = nn.Parameter(torch.empty(1, self.num_wtok, input_dim), requires_grad=True)
        # _no_grad_trunc_normal_(self.wtok, std=0.02)

        ModifiedDW_args = {
            'input_dim': self.input_dim,
            "ffn_embed_dim": ffn_embed_dim,
            "num_layers": None,
            "num_heads": num_heads,
            "num_classes": num_classes,
        }
        Encoder_args = {
            'num_layers': None, 'embed_dim': self.input_dim, 'num_heads': num_heads,
            'ffn_embed_dim': ffn_embed_dim, 'local_size': None, 
            'dropout': dropout, 'attention_dropout': attention_dropout, 
            'activation': 'relu', 'use_position': True}
        Merge_args = {
            'in_channels': self.input_dim, 'merge_scale': None, 'expand': None, 
            }
        
        self.layers = make_layers_dw_v2(
            Locals, Merge, expand, num_layers, num_layers_modified_dw,
            ModifiedDWBlock, SpeechDW_EncoderBlock, SpeechDW_MergeBlock,
            ModifiedDW_args, Encoder_args, Merge_args
            )
        ### 31/03 fix end: remove initialized wtok
        
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        dim_expand = abs(reduce(lambda x, y: x * y, expand))
        classifier_dim = self.input_dim * dim_expand
        
        assert classifier in ["Dense", "SVM", "RandomForest", "LogisticRegression"]
        
        ### 09/04/2024 fix start
        if classifier == "Dense":
            self.classifier = nn.Sequential(
                nn.Linear(classifier_dim, classifier_dim//2),
                nn.ReLU(True),
                nn.Dropout(0.5),
                nn.Linear(classifier_dim//2, classifier_dim//4),
                nn.ReLU(True),
                nn.Dropout(0.5),
                nn.Linear(classifier_dim//4, num_classes),
            )
        elif classifier == "SVM":
            from module.classifier_layer import SVM_Classifier
            self.classifier = SVM_Classifier(classifier_dim, self.input_dim, num_classes)
        elif classifier == "RandomForest":
            pass
        elif classifier == "LogisticRegression":
            pass
        else:
            return
        ### 09/04/2024 fix end

    def forward(self, x):
        if self.input_dim != x.shape[-1]:
            x = x[:, :, :self.input_dim]

        ### 31/03 fix start: remove initialized wtok
        # wtok = self.wtok.expand(x.shape[0], -1, -1)
        # x = torch.cat((wtok, x), dim=1)
        ### 31/03 fix start: remove initialized wtok

        x = self.layers(x)
        if isinstance(x, tuple):
            x = x[0]
        x = self.avgpool(x.transpose(-1, -2)).squeeze(dim=-1)

        ### 09/04/2024 fix start
        pred = self.classifier(x)
        
        return pred
        ### 09/04/2024 fix end

def make_layers_dw(Locals: list, Merge: list, expand: list, 
                   num_layers: list, num_layers_modified_dw: list,
                   ModifiedDW_blocks, Encoder_blocks, Merge_blocks, 
                   ModifiedDW_args: dict, Encoder_args: dict, Merge_args: dict):
    layers = []
    last_merge = 1
    while len(expand) < len(Merge):
        expand = expand + [-1]

    for l, ms, exp, num_encoder, num_dw in zip(Locals, Merge, expand, 
                                               num_layers, num_layers_modified_dw):
        _l = l // last_merge if l != -1 else -1
        _ms = ms // last_merge if ms != -1 else -1

        # ModifiedDW block
        ModifiedDW_args["num_layers"] = num_dw
        module1 = ModifiedDW_blocks(**ModifiedDW_args)
        layers += [module1]
        
        # Encoder block
        Encoder_args['local_size'] = _l
        module2 = Encoder_blocks(Encoder_args, num_layers=num_encoder)
        layers += [module2]

        # Merge block
        if Merge_blocks is not None:
            if _ms != -1:
                Merge_args['merge_scale'] = _ms
                Merge_args['expand'] = exp
                module3 = Merge_blocks(**Merge_args)
                layers += [module3]

                ModifiedDW_args["input_dim"] *= exp
                ModifiedDW_args["ffn_embed_dim"] *= exp

                Encoder_args['embed_dim'] *= exp
                Encoder_args['ffn_embed_dim'] *= exp
        
                Merge_args['in_channels'] *= exp
            last_merge = ms
        
        if Encoder_args['use_position']:
            Encoder_args['use_position'] = False   # only the first layer use positional embedding.
            
    return nn.Sequential(*layers)

def make_layers_dw_v2(Locals: list, Merge: list, expand: list, 
                   num_layers: list, num_layers_modified_dw: list,
                   ModifiedDW_blocks, Encoder_blocks, Merge_blocks, 
                   ModifiedDW_args: dict, Encoder_args: dict, Merge_args: dict):
    layers = []
    last_merge = 1
    while len(expand) < len(Merge):
        expand = expand + [-1]

    ModifiedDW_args["num_layers"] = 2
    module1 = ModifiedDW_blocks(**ModifiedDW_args)
    
    for l, ms, exp, num_encoder in zip(Locals, Merge, expand, num_layers):
        _l = l // last_merge if l != -1 else -1
        _ms = ms // last_merge if ms != -1 else -1

        # ModifiedDW block
        layers += [module1]
        
        # Encoder block
        Encoder_args['local_size'] = _l
        module2 = Encoder_blocks(Encoder_args, num_layers=num_encoder)
        layers += [module2]

        # Merge block
        if Merge_blocks is not None:
            if _ms != -1:
                Merge_args['merge_scale'] = _ms
                Merge_args['expand'] = exp
                module3 = Merge_blocks(**Merge_args)
                layers += [module3]

                ModifiedDW_args["input_dim"] *= exp
                ModifiedDW_args["ffn_embed_dim"] *= exp

                Encoder_args['embed_dim'] *= exp
                Encoder_args['ffn_embed_dim'] *= exp
        
                Merge_args['in_channels'] *= exp
            last_merge = ms
        
        if Encoder_args['use_position']:
            Encoder_args['use_position'] = False   # only the first layer use positional embedding.
            
    return nn.Sequential(*layers)
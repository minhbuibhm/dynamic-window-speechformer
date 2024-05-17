import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from module.utils import _get_activation_fn, add_position, get_overlap_segments
from loguru import logger

class Speech_MSA_DW(nn.Module):
    ''' The Multi-Head Self-Attention in SpeechFormer++.
    '''
    def __init__(self, embed_dim, num_heads, local_size, dropout=0., bias=True, 
                #  num_wtok=0, window_mapping = None
                 ):
        '''
        Args: 
            num_wtok: the number of learnable word tokens.
        '''
        super(Speech_MSA_DW, self).__init__()
        self.qdim = embed_dim
        self.kdim = embed_dim
        self.vdim = embed_dim
        self.local_size = local_size

        # assert num_wtok > 0 or window_mapping is not None 
        # max_num_windows = num_wtok
        # self.window_mapping = window_mapping

        self.project_qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.project_out = nn.Linear(embed_dim, embed_dim, bias=bias)

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.head_dim = embed_dim // num_heads
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.dropout = dropout
        self.scaling = float(self.head_dim) ** -0.5

    def forward(self, x, window_mapping):
        '''
        Args:
            key_padding_mask: if provided, specified padding elements in the key will
                be ignored by the attention. This is an binary mask. When the value is True,
                the corresponding value on the attention layer will be filled with -inf.
            attn_mask: mask that prevents attention to certain positions. This is an additive mask
                (i.e. the values will be added to the attention layer).
        Shape:
            Inputs:
            - x: :math:`(B, T, E)` where T is the target sequence length, B is the batch size, E is
              the embedding dimension.
            - key_padding_mask: :math:`(B, S)`, ByteTensor, where B is the batch size, S is the source sequence length.
              3-D key_padding_mask with math:`(B, T, S)` is supported now, where T is the target sequence length.
            - attn_mask: :math:`(T, S)` where T is the target sequence length, S is the source sequence length.
        '''
        b, t = x.shape[:2]
        max_num_windows, num_frames = window_mapping.shape[1:]
        global_attn = self.local_size == -1

        Q, K, V = self.project_qkv(x).chunk(3, dim=-1)
        # x: (bsz, num_wtok + tsz, fea_dim)
                
        # 28/3 add start
        # (b * self.num_heads, max_num_windows, self.head_dim)
        dw_output_wtok = x[:, : max_num_windows].transpose(0, 1).reshape(max_num_windows, b * self.num_heads, self.head_dim).transpose(0, 1)
        # 28/3 add end
        
        Q = Q * self.scaling
        Q = Q.transpose(0, 1).reshape(t, b * self.num_heads, self.head_dim).transpose(0, 1)
        K = K.transpose(0, 1).reshape(t, b * self.num_heads, self.head_dim).transpose(0, 1)
        V = V.transpose(0, 1).reshape(t, b * self.num_heads, self.head_dim).transpose(0, 1)
        # x: (bsz, num_wtok + tsz, fea_dim)
        # Q: (bsz * num_heads, num_wtok + tsz, head_dim)
        if not global_attn:
            Q_wtok, Q = Q[:, :max_num_windows], Q[:, max_num_windows:]
            K_wtok, K = K[:, :max_num_windows], K[:, max_num_windows:]
            V_wtok, V = V[:, :max_num_windows], V[:, max_num_windows:]
            # Q_wtok:   (bsz * num_heads, num_wtok, head_dim)
            # Q:        (bsz * num_heads, tsz, head_dim)
            
            t_fea = Q.shape[1]                  # t_fea = tsz

            # Compute the learnable word tokens first, then the feature tokens.
            pad_fea = (t_fea % max_num_windows)   # tsz % num_wtok
            
            if pad_fea:
                pad_len = max_num_windows - pad_fea
                # last dimension: pad 0 elements on left and right
                # second last dimension: pad 0 elements on left and pad_len elements on right
                K_pad = F.pad(K, (0, 0, 0, pad_len), value=0)
                V_pad = F.pad(V, (0, 0, 0, pad_len), value=0)
                # K_pad, V_pad: (bsz * num_heads, tsz + pad_len, head_dim)
            else:
                pad_len = 0
                K_pad = K
                V_pad = V
            ################## 30/3 fix start: replace calculated wtok by input wtok
            
            # K_pad = K_pad.reshape(b * self.num_heads, max_num_windows, -1, self.head_dim)
            # V_pad = V_pad.reshape(b * self.num_heads, max_num_windows, -1, self.head_dim)
            # # K_pad, V_pad: 
            # # (bsz * num_heads, num_wtok, (tsz + pad_len) / num_wtok, head_dim)
            
            # attn_weights_wtok = torch.matmul(
            #     Q_wtok.unsqueeze(dim=2),
            #     # (bsz * num_heads, num_wtok, 1, head_dim)
            #     torch.cat((K_wtok.unsqueeze(dim=2), K_pad), dim=2).transpose(-1, -2)
            #     # (bsz * num_heads, num_wtok, head_dim, 1 + (tsz + pad_len) / num_wtok)
            #     )
            # attn_weights_wtok = F.softmax(attn_weights_wtok, dim=-1)
            # attn_weights_wtok = F.dropout(attn_weights_wtok, p=self.dropout, training=self.training)
            # # attn_weights_wtok: (bsz * num_heads, num_wtok, 1, 1 + (tsz + pad_len) / num_wtok)

            # output_wtok = torch.matmul(
            #     attn_weights_wtok, 
            #     # (bsz * num_heads, num_wtok, 1, 1 + (tsz + pad_len) / num_wtok)
            #     torch.cat((V_wtok.unsqueeze(dim=2), V_pad), dim=2)
            #     # (bsz * num_heads, num_wtok, 1 + (tsz + pad_len) / num_wtok, head_dim)
            #     ).squeeze(dim=2)
            # # output_wtok: (bsz * num_heads, num_wtok, head_dim)
            ################## 30/3 fix end: replace calculated wtok by input wtok
            
            # TODO: expand window_mapping 
            window_mapping = window_mapping[:, :, None, :]\
                .expand(-1, -1, self.num_heads, -1)\
                    .transpose(1,2)\
                        .reshape(b * self.num_heads, max_num_windows, num_frames)
            # (bsz, max_num_windows, num_frames) 
            # -> (bsz, max_num_windows, 1, num_frames) 
            # -> (bsz, max_num_windows, num_heads, num_frames) 
            # -> (bsz, num_heads, max_num_windows, num_frames) 
            # -> (bsz * num_heads, max_num_windows, num_frames)
            
            dw_output_wtok_expa = torch.matmul(
                dw_output_wtok.transpose(1,2),
                # (bsz * num_heads, num_wtok, head_dim)
                # -> (bsz * num_heads, head_dim, num_wtok)
                window_mapping
                # (bsz * num_heads, max_num_windows, )
                ).transpose(1,2)
            # dw_output_wtok_expa: (bsz * num_heads, num_frames, head_dim)
            
            '''
            - speechformer2: complete calculate word tokens
            - now expand wtok and calc output x
            '''
            # 30/3 fix start: replace expand_wtok
            # expand = math.ceil(t_fea / max_num_windows) # tsz / num_wtok = 10 / 3 = 4
            # output_wtok_expa = output_wtok[:, :, None, :]\
            #     .expand(-1, -1, expand, -1)\
            #         .reshape(b * self.num_heads, -1, self.head_dim)\
            #             [:, :t_fea]
            output_wtok_expa = dw_output_wtok_expa
            # 30/3 fix end
            ''' 
            expand feature dimension of wtok = tsz
            output_wtok -> output_wtok_expa: 
            output_wtok:        (bsz * num_heads, num_wtok, head_dim)
            [:, :, None, :] ->  (bsz * num_heads, num_wtok, 1, head_dim)
            .expand ->          (bsz * num_heads, num_wtok, ceil(tsz / num_wtok), head_dim)
            .reshape ->         (bsz * num_heads, num_wtok * ceil(tsz / num_wtok), head_dim)
            [:, :t_fea]         (bsz * num_heads, tsz, head_dim)
            '''
            # K: (bsz * num_heads, tsz, head_dim)
            K = get_overlap_segments(K, window_size=self.local_size)
            V = get_overlap_segments(V, window_size=self.local_size)
            # K: (bsz * num_heads, tsz, window_size, head_dim) ??????

            attn_weights_fea = torch.matmul(
                Q.unsqueeze(dim=2), 
                # Q: (bsz * num_heads, tsz, head_dim)
                # -> (bsz * num_heads, tsz, 1, head_dim)
                torch.cat((output_wtok_expa.unsqueeze(dim=2), K), dim=-2).transpose(-1, -2)
            )
            ''' torch.cat((output_wtok_expa.unsqueeze(dim=2), K), dim=-2).transpose(-1, -2):
            (bsz * num_heads, num_wtok * ceil(tsz / num_wtok), head_dim)
            (bsz * num_heads, num_wtok * ceil(tsz / num_wtok), 1, head_dim)
            (bsz * num_heads, num_wtok * ceil(tsz / num_wtok), 1 + window_size, head_dim)
            (bsz * num_heads, tsz, head_dim, 1 + window_size)
            '''
            # attn_weights_fea: (bsz * num_heads, tsz, 1, 1 + window_size)
            # Separate softmax operations
            weights_wtok, weights_fea = attn_weights_fea[:, :, :, :1], attn_weights_fea[:, :, :, 1:]
            # weights_wtok: (bsz * num_heads, tsz, 1, 1)
            # weights_fea: (bsz * num_heads, tsz, 1, window_size)
            
            weights_wtok = weights_wtok.reshape(b * self.num_heads, t_fea)
            # weights_wtok: (bsz * num_heads, tsz)
            
            if pad_len:
                weights_wtok = F.pad(weights_wtok, (0, pad_len), value=float('-inf'))
                # weights_wtok: (bsz * num_heads, tsz + pad_len)
                
            weights_wtok = weights_wtok.reshape(b * self.num_heads, max_num_windows, -1)
            # weights_wtok: (bsz * num_heads, num_wtok, local_size)
            
            weights_wtok = F.softmax(weights_wtok, dim=-1).reshape(b * self.num_heads, -1)[:, :t_fea, None, None]
            '''
            softmax(weights_wtok, dim=-1):      (bsz * num_heads, num_wtok, local_size)
            .reshape(b * self.num_heads, -1):   (bsz * num_heads, tsz + pad_len)
            [:, :t_fea, None, None]             (bsz * num_heads, tsz, 1, 1)
            '''
            
            weights_fea = F.softmax(weights_fea, dim=-1)
            # weights_fea: (bsz * num_heads, tsz, 1, window_size)
            
            attn_weights_fea = torch.cat((weights_wtok, weights_fea), dim=-1)
            # attn_weights_fea: (bsz * num_heads, tsz, 1, 1 + window_size)
            
            attn_weights_fea = F.dropout(attn_weights_fea, p=self.dropout, training=self.training)
            output_fea = torch.matmul(
                attn_weights_fea, 
                # (bsz * num_heads, tsz, 1, 1 + window_size)
                torch.cat((output_wtok_expa.unsqueeze(dim=2), V), dim=-2)
                # output_wtok_expa:     (bsz * num_heads, tsz, head_dim)
                # .unsqueeze(dim=2):    (bsz * num_heads, tsz, 1, head_dim)
                # cat V:    (bsz * num_heads, tsz, 1 + window_size, head_dim)
            ).squeeze(dim=2)
            # output_fea:       (bsz * num_heads, tsz, 1, head_dim)
            # .squeeze(dim=2):  (bsz * num_heads, tsz, head_dim)
            
            # 30/3 fix start: return only num_frames, not num_wtok
            # out = torch.cat([output_wtok, output_fea], dim=1)
            # # output_wtok:  (bsz * num_heads, num_wtok, head_dim)
            # # output_fea:   (bsz * num_heads, tsz, head_dim)
            # # out:          (bsz * num_heads, num_wtok + tsz, head_dim)
            out = torch.cat([dw_output_wtok, output_fea], dim=1)
            # out = output_fea
            # 30/3 fix start: return only num_frames, not num_wtok
            
        else:
            attn_weights = torch.matmul(Q, K.transpose(-1, -2))
            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

            out = torch.matmul(attn_weights, V)
            
        # 30/3 fix start: return only num_frames, not num_wtok
        # out = out.transpose(0, 1).reshape(num_frames, b, self.embed_dim).transpose(0, 1)
        out = out.transpose(0, 1).reshape(t, b, self.embed_dim).transpose(0, 1)
        # out:                              (bsz * num_heads, num_wtok + tsz, head_dim)
        # .transpose(0, 1):                 (num_wtok + tsz, bsz * num_heads, head_dim)
        # .reshape(t, b, self.embed_dim):   (num_wtok + tsz, bsz, head_dim * num_heads)
        # .transpose(0, 1):                 (bsz, num_wtok + tsz, head_dim * num_heads)
        # 30/3 fix end: return only num_frames, not num_wtok
        
        out = self.project_out(out) #       (bsz, num_wtok + tsz, head_dim * num_heads)

        return out
    
class SpeechDW_UnitEncoder(nn.Module):
    def __init__(self, embed_dim=1024, ffn_embed_dim=2304, local_size=0, 
                 num_heads=8, dropout=0.1, attention_dropout=0.1, 
                 activation='relu', overlap=True, 
                #  num_wtok=0, window_mapping = None
                 ) -> None:
        super().__init__()
        self.dropout = dropout
        self.activation_fn = _get_activation_fn(activation)

        self.attention = Speech_MSA_DW(embed_dim, num_heads, local_size, attention_dropout, 
                                    #    num_wtok=num_wtok, window_mapping=window_mapping
                                       )

        self.attention_layer_norm = nn.LayerNorm(embed_dim)
        
        self.fc1 = nn.Linear(embed_dim, ffn_embed_dim)
        self.fc2 = nn.Linear(ffn_embed_dim, embed_dim)
        self.final_layer_norm = nn.LayerNorm(embed_dim)
    def forward(self, x, window_mapping, x_position=None):
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer implementation.
        """
        residual = x
        x = add_position(x, x_position)

        x = self.attention(x, window_mapping)  # kmeans_mask=kmeans_mask
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.attention_layer_norm(x)

        residual = x
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.final_layer_norm(x)
        return x

import sys

import copy

import torch
import numpy as np
from einops import rearrange
from typing import Optional, Tuple, Union

from torch import nn
from transformers import CLIPModel as HFCLIPModel, CLIPVisionConfig
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.clip.modeling_clip import CLIP_VISION_INPUTS_DOCSTRING,CLIPVisionTransformer
from transformers.utils import replace_return_docstrings, add_start_docstrings_to_model_forward
from model.process_clip import get_global_value, set_global_value


def SET_GLOBAL_VALUE(k, v):
    set_global_value(k, v)

class CLIPVisionEmbeddings(nn.Module):
    def __init__(self, config: CLIPVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.class_embedding = nn.Parameter(torch.randn(self.embed_dim))

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer("position_ids", torch.arange(self.num_positions).expand((1, -1)))

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        # (b t) c h w
        batch_size = pixel_values.shape[0]
        patch_embeds = self.patch_embedding(pixel_values)  # shape = [*, width, grid, grid]
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

        class_embeds = self.class_embedding.expand(batch_size, 1, -1)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        embeddings = embeddings + self.position_embedding(self.position_ids)  # b hw c
        return embeddings

class CLIPVisionEmbeddings3D(nn.Module):
    def __init__(self, config: CLIPVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.num_frames = config.num_frames
        self.tube_size = config.tube_size

        self.class_embedding = nn.Parameter(torch.randn(self.embed_dim))

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer("position_ids", torch.arange(self.num_positions).expand((1, -1)))

    def expand3d(self):

        state_dict = self.patch_embedding.state_dict()
        state_dict_expand = state_dict['weight'].unsqueeze(2)
        device, dtype = state_dict_expand.device, state_dict_expand.dtype
        # print(device, dtype)

        zero = torch.zeros_like(state_dict_expand).to(device=device, dtype=dtype)
        state_dict_expand3d = torch.cat([state_dict_expand] + (self.tube_size-1)*[zero], dim=2)

        # state_dict_expand3d = torch.cat([state_dict_expand / self.tube_size] * self.tube_size, dim=2)

        patch_embedding = nn.Conv3d(
            in_channels=self.patch_embedding.in_channels,
            out_channels=self.embed_dim,
            kernel_size=(self.tube_size, self.patch_size, self.patch_size),
            stride=(self.tube_size, self.patch_size, self.patch_size),
            bias=False,
        ).to(device=device, dtype=dtype)
        patch_embedding.load_state_dict({'weight': state_dict_expand3d})
        self.patch_embedding = patch_embedding


        class_embedding = nn.Parameter(self.class_embedding.data.repeat(self.num_frames // self.tube_size, 1)).to(device=device, dtype=dtype)
        self.class_embedding = class_embedding


    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        # (b t) c h w
        batch_size = pixel_values.shape[0] // self.num_frames
        pixel_values = rearrange(pixel_values, '(b t) c h w -> b c t h w', b=batch_size, t=self.num_frames)
        # print('pixel_values', pixel_values.shape)
        patch_embeds = self.patch_embedding(pixel_values)  # shape = [*, width, t, grid, grid]
        # print('patch_embeds', patch_embeds.shape)
        # SET_GLOBAL_VALUE('NUM_FRAMES', patch_embeds.shape[2])
        patch_embeds = rearrange(patch_embeds, 'b c t h w -> b t (h w) c')

        class_embeds = self.class_embedding.unsqueeze(1).unsqueeze(0).repeat(batch_size, 1, 1, 1)  # b t 1 c
        # print('class_embeds', class_embeds.device, class_embeds.dtype)
        # print('patch_embeds', patch_embeds.device, patch_embeds.dtype)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=2)  # b t hw+1 c
        embeddings = embeddings + self.position_embedding(self.position_ids)
        embeddings = rearrange(embeddings, 'b t hw_1 c -> (b t) hw_1 c')
        return embeddings

class CLIPModel(HFCLIPModel):
    def __init__(self, config, num_frames, add_time_attn, tube_size):
        super(CLIPModel, self).__init__(config)
        config.vision_config.num_frames = num_frames
        config.vision_config.tube_size = tube_size
        if add_time_attn:
            self.vision_model.embeddings = CLIPVisionEmbeddings(config.vision_config)
        self.T = config.vision_config.num_frames // config.vision_config.tube_size
        self.vision_model.forward = self.vision_model_forward

    @add_start_docstrings_to_model_forward(CLIP_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=CLIPVisionConfig)
    def vision_model_forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        """
        output_attentions = output_attentions if output_attentions is not None else self.vision_model.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.vision_model.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.vision_model.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        if len(pixel_values.shape) == 7:
            b_new, pair_new, T, bs_new, channel_new, h_new, w_new = pixel_values.shape
            # print(pixel_values.shape)
            B = b_new * pair_new * bs_new
            pixel_values = pixel_values.reshape(B*T, channel_new, h_new, w_new)

        elif len(pixel_values.shape) == 5:
            B, _, T, _, _ = pixel_values.shape
            # print(pixel_values.shape)
            pixel_values = rearrange(pixel_values, 'b c t h w -> (b t) c h w')
        else:
            # print(pixel_values.shape)
            B, _, _, _ = pixel_values.shape
            T = 1
        hidden_states = self.vision_model.embeddings(pixel_values)
        # print('hidden_states', hidden_states.shape)
        #
        # if self.temporal_embedding is not None and get_global_value()['NUM_FRAMES'] != 1:
        #     n = hidden_states.shape[1]
        #     hidden_states = rearrange(hidden_states, '(b t) n d -> (b n) t d', t=T)
        #     hidden_states = hidden_states + self.temporal_embedding[:, :T, :]
        #     hidden_states = rearrange(hidden_states, '(b n) t d -> (b t) n d', n=n)
        T = self.T
        # print('B.shape, T.shape', B.shape, T.shape)
        hidden_states = self.vision_model.patch_dropout(hidden_states, B, T)
        # print('patch_dropout', hidden_states.shape)
        hidden_states = self.vision_model.pre_layrnorm(hidden_states)

        encoder_outputs = self.vision_model.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        pooled_output = last_hidden_state[:, 0, :]
        pooled_output = self.vision_model.post_layernorm(pooled_output)

        pooled_output = pooled_output.reshape(B, T, -1).mean(1)
        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    def encode_image(self, image, normalize: bool = False):
        vision_outputs = self.vision_model(
            pixel_values=image,
            return_dict=True,
        )
        image_embeds = vision_outputs[1]
        image_embeds = self.visual_projection(image_embeds)
        return image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True) if normalize else image_embeds

    def encode_text(self, input_ids, attention_mask, normalize: bool = False):
        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        text_embeds = text_outputs[1]
        text_embeds = self.text_projection(text_embeds)
        return text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True) if normalize else text_embeds


    def forward(
            self,
            image=None,
            input_ids=None, attention_mask=None
    ):
        image_features = self.encode_image(image, normalize=True) if image is not None else None
        text_features = self.encode_text(input_ids, attention_mask, normalize=True) if input_ids is not None else None
        # if self.output_dict:
        return {
            "image_features": image_features,
            "text_features": text_features,
            "logit_scale": self.logit_scale.exp()
        }
        # return image_features, text_features, self.logit_scale.exp()

class TLVModel(HFCLIPModel):
    def __init__(self, args, config, num_frames, add_time_attn, tube_size):
        super(TLVModel, self).__init__(config)

        config.vision_config.num_frames = num_frames
        config.vision_config.tube_size = tube_size

        self.touch_model = CLIPVisionTransformer(config.vision_config)
        self.touch_projection = nn.Linear(config.vision_config.hidden_size, self.projection_dim, bias=False)

        if add_time_attn:
            self.touch_model.embeddings = CLIPVisionEmbeddings(config.vision_config)
            self.vision_model.embeddings = CLIPVisionEmbeddings(config.vision_config)
        self.T = config.vision_config.num_frames // config.vision_config.tube_size
        self.touch_model.forward = self.touch_model_forward
        self.vision_model.forward = self.vision_model_forward

        


    @add_start_docstrings_to_model_forward(CLIP_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=CLIPVisionConfig)
    def touch_model_forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        """
        output_attentions = output_attentions if output_attentions is not None else self.touch_model.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.touch_model.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.touch_model.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        if len(pixel_values.shape) == 7:
            b_new, pair_new, T, bs_new, channel_new, h_new, w_new = pixel_values.shape
            # print(pixel_values.shape)
            B = b_new * pair_new * bs_new
            pixel_values = pixel_values.reshape(B*T, channel_new, h_new, w_new)

        elif len(pixel_values.shape) == 5:
            B, _, T, _, _ = pixel_values.shape
            # print(pixel_values.shape)
            pixel_values = rearrange(pixel_values, 'b c t h w -> (b t) c h w')
        else:
            # print(pixel_values.shape)
            B, _, _, _ = pixel_values.shape
            T = 1
        # print('111==>', pixel_values.shape)
        hidden_states = self.touch_model.embeddings(pixel_values)
        #print('hidden_states', hidden_states.shape)
        #
        # if self.temporal_embedding is not None and get_global_value()['NUM_FRAMES'] != 1:
        #     n = hidden_states.shape[1]
        #     hidden_states = rearrange(hidden_states, '(b t) n d -> (b n) t d', t=T)
        #     hidden_states = hidden_states + self.temporal_embedding[:, :T, :]
        #     hidden_states = rearrange(hidden_states, '(b n) t d -> (b t) n d', n=n)
        T = self.T
        # print('B.shape, T.shape', B.shape, T.shape)
        hidden_states = self.touch_model.patch_dropout(hidden_states, B, T)
        # print('patch_dropout', hidden_states.shape)
        hidden_states = self.touch_model.pre_layrnorm(hidden_states)

        encoder_outputs = self.touch_model.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        pooled_output = last_hidden_state[:, 0, :]
        pooled_output = self.touch_model.post_layernorm(pooled_output)

        pooled_output = pooled_output.reshape(B, T, -1).mean(1)
        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
    def vision_model_forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        """
        output_attentions = output_attentions if output_attentions is not None else self.vision_model.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.vision_model.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.vision_model.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        if len(pixel_values.shape) == 7:
            b_new, pair_new, T, bs_new, channel_new, h_new, w_new = pixel_values.shape
            # print(pixel_values.shape)
            B = b_new * pair_new * bs_new
            pixel_values = pixel_values.reshape(B*T, channel_new, h_new, w_new)

        elif len(pixel_values.shape) == 5:
            B, _, T, _, _ = pixel_values.shape
            # print(pixel_values.shape)
            pixel_values = rearrange(pixel_values, 'b c t h w -> (b t) c h w')
        else:
            # print(pixel_values.shape)
            B, _, _, _ = pixel_values.shape
            T = 1
        hidden_states = self.vision_model.embeddings(pixel_values)
        # print('hidden_states', hidden_states.shape)
        #
        # if self.temporal_embedding is not None and get_global_value()['NUM_FRAMES'] != 1:
        #     n = hidden_states.shape[1]
        #     hidden_states = rearrange(hidden_states, '(b t) n d -> (b n) t d', t=T)
        #     hidden_states = hidden_states + self.temporal_embedding[:, :T, :]
        #     hidden_states = rearrange(hidden_states, '(b n) t d -> (b t) n d', n=n)
        T = self.T
        # print('B.shape, T.shape', B.shape, T.shape)
        hidden_states = self.vision_model.patch_dropout(hidden_states, B, T)
        # print('patch_dropout', hidden_states.shape)
        hidden_states = self.vision_model.pre_layrnorm(hidden_states)

        encoder_outputs = self.vision_model.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        pooled_output = last_hidden_state[:, 0, :]
        pooled_output = self.vision_model.post_layernorm(pooled_output)

        pooled_output = pooled_output.reshape(B, T, -1).mean(1)
        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    def encode_touch_rm_proj(self, touch, normalize: bool = False):
        touch_outputs = self.touch_model(
            pixel_values=touch,
            return_dict=True,
        )
        touch_embeds = touch_outputs[1]
        return touch_embeds / touch_embeds.norm(p=2, dim=-1, keepdim=True) if normalize else touch_embeds
    
    def encode_feeling_touch_rm_proj(self, touch, normalize: bool = False):
        gelsightA_before = touch[:,:3,:,:]
        gelsightA_after = touch[:,3:6,:,:]
        gelsightB_before = touch[:,6:9,:,:]
        gelsightB_after = touch[:,9:12,:,:]

        gelsightA_before_outputs = self.touch_model(
            pixel_values=gelsightA_before,
            return_dict=True,
        )
        gelsightA_after_outputs = self.touch_model(
            pixel_values=gelsightA_after,
            return_dict=True,
        )
        gelsightB_before_outputs = self.touch_model(
            pixel_values=gelsightB_before,
            return_dict=True,
        )
        gelsightB_after_outputs = self.touch_model(
            pixel_values=gelsightB_after,
            return_dict=True,
        )

        gelsightA_before_embeds = gelsightA_before_outputs[1]
        gelsightA_after_embeds = gelsightA_after_outputs[1]
        gelsightB_before_embeds = gelsightB_before_outputs[1]
        gelsightB_after_embeds = gelsightB_after_outputs[1]
        

        gelsightA_before_feat = gelsightA_before_embeds / gelsightA_before_embeds.norm(p=2, dim=-1, keepdim=True) if normalize else gelsightA_before_embeds
        gelsightA_after_feat = gelsightA_after_embeds / gelsightA_after_embeds.norm(p=2, dim=-1, keepdim=True) if normalize else gelsightA_after_embeds
        gelsightB_before_feat = gelsightB_before_embeds / gelsightB_before_embeds.norm(p=2, dim=-1, keepdim=True) if normalize else gelsightB_before_embeds
        gelsightB_after_feat = gelsightB_after_embeds / gelsightB_after_embeds.norm(p=2, dim=-1, keepdim=True) if normalize else gelsightB_after_embeds


        # # 新加
        # gelsight = (gelsightA_before_embeds + gelsightA_after_embeds) / 2
        # gelsight = self.touch_projection(gelsight)
        # gelsight_feat = gelsight / gelsight.norm(p=2, dim=-1, keepdim=True) if normalize else gelsight
        # return gelsight_feat
    
        return gelsightA_before_feat,gelsightA_after_feat,gelsightB_before_feat,gelsightB_after_feat
    
    def encode_touch(self, touch, normalize: bool = False):
        touch_outputs = self.touch_model(
            pixel_values=touch,
            return_dict=True,
        )
        touch_embeds = touch_outputs[1]
        touch_embeds = self.touch_projection(touch_embeds)
        return touch_embeds / touch_embeds.norm(p=2, dim=-1, keepdim=True) if normalize else touch_embeds

    def encode_vision(self, vision, normalize: bool = False):
        vision_outputs = self.vision_model(
            pixel_values=vision,
            return_dict=True,
        )
        image_embeds = vision_outputs[1]
        image_embeds = self.visual_projection(image_embeds)
        return image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True) if normalize else image_embeds

    def encode_text(self, input_ids, attention_mask, normalize: bool = False):
        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        text_embeds = text_outputs[1]
        text_embeds = self.text_projection(text_embeds)
        return text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True) if normalize else text_embeds
        
    def beta_decay(self, args, total_steps, step):
        if args.decay_type == 'linear':
            beta = args.beta_init*((total_steps-step)/total_steps)
        elif args.decay_type == 'exp':
            base = np.power(0.1, 1/total_steps)
            beta = args.beta_init*np.power(base, step)
        elif args.decay_type == 'cosine':
            beta = args.beta_init*(1 + np.cos(np.pi * step / total_steps))/2
        else:
            raise NotImplementedError
        return max(beta,0)
    
    def beta_decay2(self, args, total_steps, step):
        total_step = (total_steps/args.span)*(1-args.span)
        beta = args.beta_end*((step-total_steps)/total_step)
        return -1*beta

    def learn_curriculum(self, args, touch_features, vision_features, total_steps, step):
        if args.inte_type == 'below': # beta might be less than 0
            beta = self.beta_decay(args, total_steps, step) if total_steps>=step else self.beta_decay2(args, total_steps, step)
            curriculum_representation = (1-beta)*touch_features + beta*vision_features
        elif args.inte_type == 'above': # beta must be at least 0
            beta = self.beta_decay(args, total_steps, step) if total_steps>=step else 0
            curriculum_representation = (1-beta)*touch_features + beta*vision_features
        else:
            raise NotImplementedError
        return curriculum_representation
    
    def feeling_forward(
            self, args=None, touch=None, vision=None, input_ids=None, attention_mask=None,
            total_steps=None, current_step=None
    ):
        touch_features = self.encode_touch(touch, normalize=True) if touch is not None else None
        vision_features = self.encode_vision(vision, normalize=True) if vision is not None else None
        text_features = self.encode_text(input_ids, attention_mask, normalize=True) if input_ids is not None else None
        curriculum_representation = self.learn_curriculum(args, touch_features, vision_features, total_steps, current_step)\
                                                                    if total_steps is not None else None
        return {
            "touch_features": touch_features,
            "vision_features": vision_features,
            "text_features": text_features,
            "curriculum_representation": curriculum_representation,
            "tl_logit_scale": self.tl_logit_scale.exp(),
            "tv_logit_scale": self.tv_logit_scale.exp(),
        }


    def forward(
            self, args=None, touch=None, vision=None, sent_input_ids=None, sent_attention_mask=None,
            phra_input_ids=None, phra_attention_mask=None, total_steps=None, current_step=None
    ):
        touch_features = self.encode_touch(touch, normalize=True) if touch is not None else None
        vision_features = self.encode_vision(vision, normalize=True) if vision is not None else None
        curriculum_representation = self.learn_curriculum(args, touch_features, vision_features, total_steps, current_step)\
                                                                    if total_steps is not None else None
        
        sent_features = self.encode_text(sent_input_ids, sent_attention_mask, normalize=True) if sent_input_ids is not None else None
        phra_features = self.encode_text(phra_input_ids, phra_attention_mask, normalize=True) if phra_input_ids is not None else None   
        text_features = sent_features + phra_features if ((sent_features is not None) and (phra_features is not None)) else None
        
        return {
            "touch_features": touch_features,
            "vision_features": vision_features,
            "text_features": text_features,
            "curriculum_representation": curriculum_representation,
            "tl_logit_scale": self.logit_scale.exp(),
        }

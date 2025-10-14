import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from collections import namedtuple
from transformers.models.gpt2 import GPT2LMHeadModel
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor, ChameleonProcessor
import logging
logging.basicConfig(
    filename='chameleon_sqa_infer_64_full.log',  
    level=logging.DEBUG,         
    format='[%(asctime)s] %(message)s',  
    datefmt='%Y-%m-%d %H:%M:%S'   
)
import pdb
from transformers.cache_utils import DynamicCache

Outputs = namedtuple("Outputs", ["loss", "inputs_embeds", "logits"])
MAX_N_LATENT = 4


class IVTLR(nn.Module):

    def __init__(
        self,
        base_causallm,
        latent_token_id,
        start_latent_id,
        end_latent_id,
        eos_token_id,
        image_token_id,
        num_selected_patches: int = 64,
    ):

        super(IVTLR, self).__init__()
        self.gen_forward_cnt = 0
        self.base_causallm = base_causallm
        self.latent_token_id = latent_token_id
        self.eos_token_id = eos_token_id
        self.start_latent_id = start_latent_id
        self.end_latent_id = end_latent_id
        self.image_token_id = image_token_id
        self.num_selected_patches = num_selected_patches

        # tested with GPT2 and Llama3
        if isinstance(self.base_causallm, GPT2LMHeadModel):
            self.embedding = self.base_causallm.transformer.get_input_embeddings()
        else:
            self.embedding = self.base_causallm.get_input_embeddings()
        
        self.processor = ChameleonProcessor.from_pretrained("facebook/chameleon-7b")
        # self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
    def forward(self, input_ids, attention_mask, labels, position_ids, pixel_values, **kwargs):
        all_logits = []
        B, orig_S = input_ids.size()
        device = input_ids.device
        # latent 
        latent_indices = (input_ids == self.latent_token_id).nonzero()
        latent_lists = [
            [idx[1].item() for idx in latent_indices if idx[0]==b]
            for b in range(B)
        ]
        max_n_latents = max(len(l) for l in latent_lists)

        model_dtype = next(self.base_causallm.parameters()).dtype


        pixel_values = pixel_values.to(model_dtype)

        if max_n_latents > 0:

            all_vokens = self.base_causallm.model.model.get_image_tokens(pixel_values)


            img_starts = []
            for b in range(B):
                seq = input_ids[b]
                        
                is_image_token = (seq == self.image_token_id)
                logging.debug(f"Sequence {b} length: {len(seq)}")
                
                
                img_positions = is_image_token.nonzero(as_tuple=True)[0]
                
                if img_positions.numel() > 0:  
                    first_img_index = img_positions.min().item()
                    img_starts.append(first_img_index)
                    
                is_image_token = (input_ids[b] == self.image_token_id)
        
                
                logging.debug(f"Sequence {b} length: {len(is_image_token)}")
                img_starts.append((input_ids[b] == self.image_token_id)
                                .nonzero(as_tuple=True)[0].min().item())



            new_ids = []
            new_att = []
            new_pos = []
            for b in range(B):
                start = img_starts[b]
                pre = input_ids[b, :start]
                suf = input_ids[b, start+1024:]
                
                merged = torch.cat([pre, all_vokens[b], suf], dim=0)
                new_ids.append(merged)

                #  attention_mask & position_ids
                a_pre = attention_mask[b, :start]
                a_suf = attention_mask[b, start+1024:]
                merged_a = torch.cat([a_pre, torch.ones(1024, device=device), a_suf], dim=0)
                new_att.append(merged_a)

                p_merged = torch.arange(merged.size(0), device=device)
                new_pos.append(p_merged)

            #  batch
            max_len = max(ids.size(0) for ids in new_ids)
            ids_b, att_b, pos_b = [], [], []
            for b in range(B):
                ids = new_ids[b]; att = new_att[b]; pos = new_pos[b]
                L = ids.size(0)
                
                ids_b.append(ids.unsqueeze(0))
                att_b.append(att.unsqueeze(0))
                pos_b.append(pos.unsqueeze(0))

            input_ids      = torch.cat(ids_b, dim=0)     # (B, new_S)
            attention_mask = torch.cat(att_b, dim=0)     # (B, new_S)
            position_ids   = torch.cat(pos_b, dim=0)     # (B, new_S)
            new_S = input_ids.size(1)

            inputs_embeds = self.embedding(input_ids)     # (B, new_S, D)



            original_mask = torch.ones((B, new_S), dtype=torch.bool, device=device)
            # image_mask no repeated True
            image_mask = torch.zeros((B, 3000), dtype=torch.bool, device=device)
            for b in range(B):
                s = img_starts[b]
                image_mask[b, s:s+1024] = True


            end = min((lst[0] for lst in latent_lists if lst), default=new_S)

            kv_cache = None

            for pass_idx in range(max_n_latents):
                logging.debug(f"latent num: {pass_idx}")
                start = 0
                hidden_states_offset = 0
                logging.debug(f"end:{end}")
                if kv_cache == None and pass_idx == 0:
                    outputs = self.base_causallm(
                        inputs_embeds=inputs_embeds[:, :end],
                        attention_mask=attention_mask[:, :end],
                        position_ids=position_ids[:, :end],
                        output_hidden_states=True,
                        output_attentions=True,
                        use_cache=True
                    )
                else:
                    outputs = self.base_causallm(
                        inputs_embeds=inputs_embeds[:, :end, :],
                        attention_mask=attention_mask[:, :end],
                        position_ids=position_ids[:, :end],
                        output_hidden_states=True,
                        output_attentions=True,
                        use_cache=True,
                    )
                    
                logits_this = outputs.logits
                logging.debug(f"1th logits shape: {logits_this.shape}")                      
                hidden_states = outputs.hidden_states[-1]      
                attentions    = outputs.attentions              # list of (B, heads, seq_len, seq_len)
                kv_cache      = outputs.past_key_values

                all_logits.append(logits_this)


                avg_attn = torch.cat(attentions, dim=1).mean(dim=1)  # (B, seq_len)
                current_seq_len = avg_attn.size(1)
                select_image_embeds = []

                for b in range(B):

                    last_attn = avg_attn[b, end - 1]  # shape=(seq_len,)
                    vs, ve = img_starts[b], img_starts[b]+1024
                    # current_seq_len = last_attn.size(0)

                    scores = last_attn.clone()

                    allowed_positions = image_mask[b, :current_seq_len]  # shape=(S,)
                    invalid = ~allowed_positions
                    scores[invalid] = float("-inf")


                    rel_scores = scores[vs: ve]  # (image_len,)
                    topk_rel = rel_scores.topk(self.num_selected_patches, sorted=False)[1]  # rel idx
                    abs_idxs = vs + topk_rel  
                    logging.debug(f"topk_rel: {topk_rel}")
                    logging.debug(f"abs idx: {abs_idxs}")

                    image_mask[b, abs_idxs] = False

                    picked = inputs_embeds[b, abs_idxs, :]  # (K, D)
                    select_image_embeds.append(picked)

                select_image_embeds = torch.stack(select_image_embeds, dim=0)  # (B, K, D)
                inputs_embeds_detached = inputs_embeds.detach().clone()

                for b in range(B):
                    if len(latent_lists[b]) > pass_idx:
                        t_idx = latent_lists[b][pass_idx] 
                        rel_pos = t_idx - 1 - hidden_states_offset
                        rel_pos = max(0, min(rel_pos, hidden_states.size(1) - 1))
                        inputs_embeds_detached[b, t_idx, :] = hidden_states[b, rel_pos, :]

                inputs_embeds.data = inputs_embeds_detached

                new_inputs_embeds = []
                new_attention_mask = []
                new_position_ids = []
                new_original_mask = []
                new_image_mask = []
                batch_max_len = 0

                for b in range(B):
                    end_b = end
                    prefix_b = inputs_embeds[b, :end_b, :]    # (end_b, D)
                    suffix_b = inputs_embeds[b, end_b:, :]    # (old_len - end_b, D)
                    v_embed_b = select_image_embeds[b]       # (K, D)
                    merged_b = torch.cat([prefix_b, v_embed_b, suffix_b], dim=0)  # (old_len+K, D)
                    new_inputs_embeds.append(merged_b)

                    #  attention_mask
                    att_pref = attention_mask[b, :end_b]      # (end_b,)
                    att_suf  = attention_mask[b, end_b:]      # (old_len-end_b,)
                    att_v    = torch.ones(self.num_selected_patches, device=attention_mask.device, dtype=attention_mask.dtype)
                    merged_att = torch.cat([att_pref, att_v, att_suf], dim=0)  # (new_len,)
                    new_attention_mask.append(merged_att)

                    # position_ids 
                    new_pos = torch.arange(merged_b.size(0), device=position_ids.device)
                    new_position_ids.append(new_pos)

                    # —— original_mask
                    orig_pref = original_mask[b, :end_b]       # (end_b,)
                    orig_suf  = original_mask[b, end_b:]       # (old_len-end_b,)
                    orig_v    = torch.zeros(self.num_selected_patches, device=input_ids.device, dtype=torch.bool)
                    merged_orig = torch.cat([orig_pref, orig_v, orig_suf], dim=0)
                    new_original_mask.append(merged_orig)

                    # image_mask
                    img_pref = image_mask[b, :end_b]
                    img_suf  = image_mask[b, end_b:]
                    img_v    = torch.zeros(self.num_selected_patches, device=input_ids.device, dtype=torch.bool)
                    merged_img = torch.cat([img_pref, img_v, img_suf], dim=0)
                    new_image_mask.append(merged_img)

                    batch_max_len = max(batch_max_len, merged_b.size(0))
                
                
                padded_embeds = []
                padded_att   = []
                padded_pos   = []
                padded_orig  = []
                padded_img   = []

                for b in range(B):
                    emb_b = new_inputs_embeds[b]
                    att_b = new_attention_mask[b]
                    pos_b = new_position_ids[b]
                    orig_b = new_original_mask[b]
                    img_b = new_image_mask[b]
                    
                    padded_embeds.append(emb_b.unsqueeze(0))
                    padded_att.append(att_b.unsqueeze(0))
                    padded_pos.append(pos_b.unsqueeze(0))
                    padded_orig.append(orig_b.unsqueeze(0))
                    padded_img.append(img_b.unsqueeze(0))

                inputs_embeds = torch.cat(padded_embeds, dim=0)    
                attention_mask = torch.cat(padded_att, dim=0)      
                position_ids    = torch.cat(padded_pos, dim=0)     
                original_mask  = torch.cat(padded_orig, dim=0)
                image_mask     = torch.cat(padded_img, dim=0)   # (B, new_S)

                K = self.num_selected_patches
                for b in range(B):
                    for i, pos in enumerate(latent_lists[b]):
                        if pos > end:
                            latent_lists[b][i] = pos + K
                            logging.debug(f"latent pos: {latent_lists[b][i]}")


                if pass_idx + 1 >= max_n_latents:
                    end = inputs_embeds.size(1)
                else:
                    end = end + 1 + K

               
            # final pass
            if kv_cache:
                outputs = self.base_causallm(
                    inputs_embeds=inputs_embeds[:, :end, :],
                    attention_mask=attention_mask[:, :end],
                    position_ids=position_ids[:, :end],
                    output_hidden_states=True,
                    output_attentions=False,
                )
            else:
                outputs = self.base_causallm(
                    inputs_embeds=inputs_embeds[:, :end, :],
                    attention_mask=attention_mask[:, :end],
                    position_ids=position_ids[:, :end],
                    output_hidden_states=True,
                    output_attentions=False,
                )
            all_logits.append(outputs.logits)
            logging.debug(f"final logits shape: {outputs.logits.shape}")   

        if max_n_latents == 0:
            # 4. embedding
            all_vokens = self.base_causallm.model.model.get_image_tokens(pixel_values)


            img_starts = []
            for b in range(B):
                logging.debug(f"input_ids[b]: {input_ids[b]}")
                is_image_token = (input_ids[b] == self.image_token_id)
        

                logging.debug(f"Sequence {b} length: {len(is_image_token)}")
                img_starts.append((input_ids[b] == self.image_token_id)
                                .nonzero(as_tuple=True)[0].min().item())


            new_ids = []
            new_att = []
            new_pos = []
            for b in range(B):
                start = img_starts[b]
                pre = input_ids[b, :start]
                suf = input_ids[b, start+1024:] 
                merged = torch.cat([pre, all_vokens[b], suf], dim=0)
                new_ids.append(merged)

                a_pre = attention_mask[b, :start]
                a_suf = attention_mask[b, start+1024:]
                merged_a = torch.cat([a_pre, torch.ones(1024, device=device), a_suf], dim=0)
                new_att.append(merged_a)

                p_merged = torch.arange(merged.size(0), device=device)
                new_pos.append(p_merged)

            max_len = max(ids.size(0) for ids in new_ids)
            ids_b, att_b, pos_b = [], [], []
            for b in range(B):
                ids = new_ids[b]; att = new_att[b]; pos = new_pos[b]
                L = ids.size(0)
                ids_b.append(ids.unsqueeze(0))
                att_b.append(att.unsqueeze(0))
                pos_b.append(pos.unsqueeze(0))

            input_ids_new      = torch.cat(ids_b, dim=0)     # (B, new_S)
            attention_mask_new = torch.cat(att_b, dim=0)     # (B, new_S)
            position_ids_new   = torch.cat(pos_b, dim=0)     # (B, new_S)
            new_S = input_ids.size(1)

            inputs_embeds = self.embedding(input_ids_new)     # (B, new_S, D)

            outputs = self.base_causallm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                pixel_values=pixel_values,
                output_hidden_states=True
            )
            
            all_logits.append(outputs.logits)
        
        logits = torch.cat(all_logits, dim=-2)  # (B, total_len, V)
        B, final_S, V = logits.size()


        new_labels = torch.full((B, final_S), -100, device=input_ids.device, dtype=labels.dtype)
        for b in range(B):
            num_labels = labels.size(1)
            # logging.debug(f"num_labels: {num_labels}")
            new_labels[:, -num_labels:] = labels
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = new_labels[..., 1:].contiguous()
        loss_fct = CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return Outputs(loss=loss, inputs_embeds=inputs_embeds, logits=logits)

    def train(self, mode=True):
        self.base_causallm.train(mode)

    def eval(self):
        self.base_causallm.eval()
    
    def prepare_inputs_for_generation(
            self,
            input_ids: torch.LongTensor = None,
            pixel_values: torch.FloatTensor = None,
            past_key_values: tuple = None,
            attention_mask: torch.Tensor = None,
            inputs_embeds: torch.FloatTensor = None,
            position_ids: torch.LongTensor = None,
            use_cache: bool = True,
            **kwargs
        ):
        
        self.base_causallm.prepare_inputs_for_generation(
            input_ids=input_ids,
            pixel_values=pixel_values,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            use_cache=use_cache,
            **kwargs
        )

    def generate(
        self,
        input_ids,
        attention_mask,  # attention_mask is not used
        pixel_values,
        max_new_tokens=16,
        output_embedding=False,
        synced_gpus=False,
        **kwargs
    ):
        self.gen_forward_cnt = 0
        eos_pos = None
        logging.debug(f"input_ids.shape[0]: {input_ids.shape[0]}")
        logging.debug(f"pixel_values.device: {pixel_values.device}")
        assert input_ids.shape[0] == 1, "only support batch_size == 1 now"

        tokens = input_ids[0].detach().tolist()
        print("generate 280")
        

        current_ids = input_ids.clone()
        
        position_ids = torch.arange(
            0, current_ids.shape[1], 
            dtype=torch.long, 
            device=current_ids.device
        ).reshape(1, -1)
        

        outputs = self.forward(
            input_ids=current_ids,
            attention_mask=torch.ones_like(current_ids),
            labels=current_ids.clone(), 
            position_ids=position_ids,
            pixel_values=pixel_values,
        )
        

        next_token = torch.argmax(outputs.logits[0, -1]).item()
        tokens.append(next_token)
        
        current_inputs_embeds = outputs.inputs_embeds  # shape: (1, seq_len_after_insertion, hidden_dim)
        current_seq_len = current_inputs_embeds.shape[1]
        
        current_attention_mask = torch.ones((1, current_seq_len), device=current_inputs_embeds.device)
        

        next_token_embedding = self.embedding(torch.tensor([[next_token]], device=current_inputs_embeds.device))
        current_inputs_embeds = torch.cat([current_inputs_embeds, next_token_embedding], dim=1)
        current_attention_mask = torch.cat([current_attention_mask, torch.ones((1, 1), device=current_inputs_embeds.device)], dim=1)
        
        self.gen_forward_cnt += 1
        print("generate 302")
        

        past_key_values = None
        

        for _ in range(max_new_tokens - 1):
            if past_key_values is None:
                logging.debug(f"no kv_cache, using full embedding sequence")

                inputs_embeds_for_forward = current_inputs_embeds
                attention_mask_for_forward = current_attention_mask
                position_ids_for_forward = torch.arange(
                    0, current_inputs_embeds.shape[1], 
                    dtype=torch.long, 
                    device=current_inputs_embeds.device
                ).reshape(1, -1)
            else:
                logging.debug(f"using kv_cache, input_shape: {next_token_embedding.shape}")

                inputs_embeds_for_forward = next_token_embedding
                attention_mask_for_forward = current_attention_mask
                position_ids_for_forward = torch.tensor([[current_inputs_embeds.shape[1] - 1]], device=current_inputs_embeds.device)
            

            outputs = self.base_causallm.forward(
                inputs_embeds=inputs_embeds_for_forward,
                attention_mask=attention_mask_for_forward,
                position_ids=position_ids_for_forward,
                past_key_values=past_key_values,
                use_cache=True
            )
            

            past_key_values = outputs.past_key_values
            
            self.gen_forward_cnt += 1
            
            next_token = torch.argmax(outputs.logits[0, -1]).item()
            tokens.append(next_token)

            if self.gen_forward_cnt % 10 == 0 and self.gen_forward_cnt >= 10:
                logging.debug(f"gen_forward_cnt: {self.gen_forward_cnt}")
            

            if next_token == self.eos_token_id:
                logging.debug(f"EOS token encountered at position {len(tokens)}, stopping generation")
                break
            

            next_token_embedding = self.embedding(torch.tensor([[next_token]], device=current_inputs_embeds.device))
            current_inputs_embeds = torch.cat([current_inputs_embeds, next_token_embedding], dim=1)
            current_attention_mask = torch.cat([current_attention_mask, torch.ones((1, 1), device=current_inputs_embeds.device)], dim=1)

        print("generate 315")
        
        if output_embedding:

            return torch.tensor(tokens).view(1, -1), current_inputs_embeds
        else:
            return torch.tensor(tokens).view(1, -1)

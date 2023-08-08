import torch
from typing import Optional, Tuple, Union
from torch.nn.modules import CrossEntropyLoss
from src.EnhanceBertModel import EnhanceBertModel, EnhanceBertConfig
from transformers import BertTokenizer, BertLMHeadModel, EncoderDecoderModel, PreTrainedModel, AutoModel, AutoConfig, BertConfig, EncoderDecoderConfig, logging
from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutput
from transformers.models.encoder_decoder.modeling_encoder_decoder import shift_tokens_right

logger = logging.get_logger(__name__)


class BertEncoderDecoderModel(EncoderDecoderModel):
    AutoConfig.register("EnhanceBert", EnhanceBertConfig)
    AutoModel.register(EnhanceBertConfig, EnhanceBertModel)

    @classmethod
    def from_encoder_decoder_pretrained(
        cls,
        encoder_pretrained_model_name_or_path: str = None,
        decoder_pretrained_model_name_or_path: str = None,
        *model_args,
        **kwargs
    ) -> PreTrainedModel:

        kwargs_encoder = {
            argument[len("encoder_") :]: value for argument, value in kwargs.items() if argument.startswith("encoder_")
        }

        kwargs_decoder = {
            argument[len("decoder_") :]: value for argument, value in kwargs.items() if argument.startswith("decoder_")
        }

        # remove encoder, decoder kwargs from kwargs
        for key in kwargs_encoder.keys():
            del kwargs["encoder_" + key]
        for key in kwargs_decoder.keys():
            del kwargs["decoder_" + key]

        # Load and initialize the encoder and decoder
        # The distinction between encoder and decoder at the model level is made
        # by the value of the flag `is_decoder` that we need to set correctly.
        encoder = kwargs_encoder.pop("model", None)
        if encoder is None:
            if encoder_pretrained_model_name_or_path is None:
                raise ValueError(
                    "If `encoder_model` is not defined as an argument, a `encoder_pretrained_model_name_or_path` has "
                    "to be defined."
                )

            if "config" not in kwargs_encoder:
                encoder_config, kwargs_encoder = EnhanceBertConfig.from_pretrained(
                    encoder_pretrained_model_name_or_path, **kwargs_encoder, return_unused_kwargs=True
                )

                if encoder_config.is_decoder is True or encoder_config.add_cross_attention is True:
                    logger.info(
                        f"Initializing {encoder_pretrained_model_name_or_path} as a encoder model "
                        "from a decoder model. Cross-attention and casual mask are disabled."
                    )
                    encoder_config.is_decoder = False
                    encoder_config.add_cross_attention = False
                encoder_config.architectures = "EnhanceBertModel"

                kwargs_encoder["config"] = encoder_config

            encoder = EnhanceBertModel.from_pretrained(encoder_pretrained_model_name_or_path, *model_args, **kwargs_encoder)

        decoder = kwargs_decoder.pop("model", None)
        if decoder is None:
            if decoder_pretrained_model_name_or_path is None:
                raise ValueError(
                    "If `decoder_model` is not defined as an argument, a `decoder_pretrained_model_name_or_path` has "
                    "to be defined."
                )

            if "config" not in kwargs_decoder:
                decoder_config, kwargs_decoder = BertConfig.from_pretrained(
                    decoder_pretrained_model_name_or_path, **kwargs_decoder, return_unused_kwargs=True
                )

                if decoder_config.is_decoder is False or decoder_config.add_cross_attention is False:
                    logger.info(
                        f"Initializing {decoder_pretrained_model_name_or_path} as a decoder model. Cross attention"
                        f" layers are added to {decoder_pretrained_model_name_or_path} and randomly initialized if"
                        f" {decoder_pretrained_model_name_or_path}'s architecture allows for cross attention layers."
                    )
                    decoder_config.is_decoder = True
                    decoder_config.add_cross_attention = True
                    decoder_config.architectures = "BertLMHeadModel"

                kwargs_decoder["config"] = decoder_config

            if kwargs_decoder["config"].is_decoder is False or kwargs_decoder["config"].add_cross_attention is False:
                logger.warning(
                    f"Decoder model {decoder_pretrained_model_name_or_path} is not initialized as a decoder. "
                    f"In order to initialize {decoder_pretrained_model_name_or_path} as a decoder, "
                    "make sure that the attributes `is_decoder` and `add_cross_attention` of `decoder_config` "
                    "passed to `.from_encoder_decoder_pretrained(...)` are set to `True` or do not pass a "
                    "`decoder_config` to `.from_encoder_decoder_pretrained(...)`"
                )

            decoder = BertLMHeadModel.from_pretrained(decoder_pretrained_model_name_or_path, **kwargs_decoder)

        # instantiate config with corresponding kwargs
        config = EncoderDecoderConfig.from_encoder_decoder_configs(encoder.config, decoder.config, **kwargs)
        return cls(encoder=encoder, decoder=decoder, config=config)
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        attribute_pos_ids: Optional[torch.Tensor] = None,
        num_pos_ids: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        encoder_outputs: Optional[Tuple[torch.FloatTensor]] = None,
        past_key_values: Tuple[Tuple[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, Seq2SeqLMOutput]:
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        kwargs_encoder = {argument: value for argument, value in kwargs.items() if not argument.startswith("decoder_")}

        kwargs_decoder = {
            argument[len("decoder_") :]: value for argument, value in kwargs.items() if argument.startswith("decoder_")
        }

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                attribute_pos_ids = attribute_pos_ids,
                num_pos_ids = num_pos_ids,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs_encoder,
            )
        elif isinstance(encoder_outputs, tuple):
            encoder_outputs = BaseModelOutput(*encoder_outputs)

        encoder_hidden_states = encoder_outputs[0]

        # optionally project encoder_hidden_states
        if (
            self.encoder.config.hidden_size != self.decoder.config.hidden_size
            and self.decoder.config.cross_attention_hidden_size is None
        ):
            encoder_hidden_states = self.enc_to_dec_proj(encoder_hidden_states)

        if (labels is not None) and (decoder_input_ids is None and decoder_inputs_embeds is None):
            decoder_input_ids = shift_tokens_right(
                labels, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
            past_key_values=past_key_values,
            return_dict=return_dict,
            **kwargs_decoder,
        )

        # Compute loss independent from decoder (as some shift the logits inside them)
        loss = None
        if labels is not None:
            logits = decoder_outputs.logits if return_dict else decoder_outputs[0]
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.reshape(-1, self.decoder.config.vocab_size), labels.view(-1))

        if not return_dict:
            if loss is not None:
                return (loss,) + decoder_outputs + encoder_outputs
            else:
                return decoder_outputs + encoder_outputs

        return Seq2SeqLMOutput(
            loss=loss,
            logits=decoder_outputs.logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


class BertEncoderDecoderModelFroTree(BertEncoderDecoderModel):

    def get_hidden_state(self, inputs, targets, input_posts, input_post_lengths, target_posts, num_pos, attribute_pos, tokenizer:BertTokenizer):
        # Note: we run this all at once (over multiple batches of multiple sequences)
        input_length_max = max(input_post_lengths)
        input_ids = []
        attention_mask = []
        num_pos_ids = []
        attribute_pos_ids = []

        for idx,item in enumerate(inputs):
            # Infix formal data 
            input_id = item["input_ids"].squeeze()
            mask = item["attention_mask"].squeeze()
            zeros = torch.zeros(input_length_max - input_id.size(0))
            padded = torch.cat([input_id.long(), zeros.long()])
            input_ids.append(padded)
            padded = torch.cat([mask.long(), zeros.long()])
            attention_mask.append(padded)
            if num_pos != None:
                num_pos_id = torch.ones(input_id.size(0), dtype=torch.long)
                padded = torch.cat((num_pos_id.long(), zeros.long()), dim=0).tolist()
                for num in num_pos[idx]:
                    padded[num] = 2
                padded = torch.LongTensor(padded)
                num_pos_ids.append(padded)
            if attribute_pos != None:
                attribute_pos_id = torch.ones(input_id.size(0), dtype=torch.long)
                padded = torch.cat((attribute_pos_id.long(), zeros.long()), dim=0).tolist()
                for attr in attribute_pos[idx]:
                    padded[attr] = 2
                padded = torch.LongTensor(padded)
                attribute_pos_ids.append(padded)
        for idx,item in enumerate(input_posts):
            # Postfix formal data
            input_id = item["input_ids"].squeeze()
            mask = item["attention_mask"].squeeze()
            zeros = torch.zeros(input_length_max - input_id.size(0))
            padded = torch.cat([input_id.long(), zeros.long()])
            input_ids.append(padded)
            padded = torch.cat([mask.long(), zeros.long()])
            attention_mask.append(padded)
            if num_pos != None:
                num_pos_id = torch.ones(input_id.size(0), dtype=torch.long)
                padded = torch.cat((num_pos_id.long(), zeros.long()), dim=0).tolist()
                for num in num_pos[idx]:
                    padded[num+1] = 2
                padded = torch.LongTensor(padded)
                num_pos_ids.append(padded)
            if attribute_pos != None:
                attribute_pos_id = torch.ones(input_id.size(0), dtype=torch.long)
                padded = torch.cat((attribute_pos_id.long(), zeros.long()), dim=0).tolist()
                for attr in attribute_pos[idx]:
                    padded[attr+1] = 2
                padded = torch.LongTensor(padded)
                attribute_pos_ids.append(padded)
        
        input_ids = torch.stack(input_ids, dim=0).long().cuda()
        attention_mask = torch.stack(attention_mask, dim=0).long().cuda()
        if num_pos != None:
            num_pos_ids = torch.stack(num_pos_ids, dim=0).long().cuda()
        if attribute_pos != None:    
            attribute_pos_ids = torch.stack(attribute_pos_ids, dim=0).long().cuda()

        labels = targets + target_posts
        labels = tokenizer(labels, return_tensors="pt", add_special_tokens=False, padding=True)["input_ids"].cuda()

        if attribute_pos != None and num_pos != None:
            outputs = self.forward(input_ids = input_ids, attention_mask=attention_mask, num_pos_ids=num_pos_ids, attribute_pos_ids=attribute_pos_ids, labels=labels)
        elif num_pos != None:
            outputs = self.forward(input_ids = input_ids, attention_mask=attention_mask, num_pos_ids=num_pos_ids, attribute_pos_ids=None, labels=labels)
        else:
            outputs = self.forward(input_ids = input_ids, attention_mask=attention_mask, num_pos_ids=None, attribute_pos_ids=None, labels=labels)
        last_hidden_state = outputs["encoder_last_hidden_state"]
        encoder_output = last_hidden_state[:len(input_post_lengths)].transpose(0, 1)
        loss = outputs["loss"]
        problem_output = encoder_output.mean(0)

        return encoder_output, problem_output, loss

    def evaluate(self, input, num_pos, attribute_pos):
        input_ids = input["input_ids"].long().cuda()
        attention_mask = input["attention_mask"].long().cuda()
        if num_pos != None:
            num_pos_ids = torch.ones(input["input_ids"].squeeze().size(0), dtype=torch.long).tolist()
            for num in num_pos:
                num_pos_ids[num] = 2
            num_pos_ids = torch.LongTensor(num_pos_ids).unsqueeze(0).cuda()
        if attribute_pos != None:
            attribute_pos_ids = torch.ones(input["input_ids"].squeeze().size(0), dtype=torch.long).tolist()
            for attr in attribute_pos:
                attribute_pos_ids[attr] = 2
            attribute_pos_ids = torch.LongTensor(attribute_pos_ids).unsqueeze(0).cuda()
        
        if attribute_pos != None and num_pos != None:
            output = self.encoder(input_ids, attention_mask, num_pos_ids=num_pos_ids, attribute_pos_ids=attribute_pos_ids)[0].transpose(0, 1)
        elif num_pos != None:
            output = self.encoder(input_ids, attention_mask, num_pos_ids=num_pos_ids, attribute_pos_ids=None)[0].transpose(0, 1)
        else:
            output = self.encoder(input_ids, attention_mask, num_pos_ids=None, attribute_pos_ids=None)[0].transpose(0, 1)
        problem_output = output.mean(0)

        return output, problem_output
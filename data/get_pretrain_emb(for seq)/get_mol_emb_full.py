from transformers import PreTrainedModel, RobertaModel, RobertaConfig,AutoModelWithLMHead
from collections import OrderedDict
import torch
import os
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel
from transformers import RobertaTokenizerFast,AutoTokenizer
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
class RobertaForSequenceClassification(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = ["position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        """
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return outputs
def prune_state_dict(model_dir):
    """Remove problematic keys from state dictionary"""
    if not (model_dir and os.path.exists(os.path.join(model_dir, "pytorch_model.bin"))):
        return None

    state_dict_path = os.path.join(model_dir, "pytorch_model.bin")
    assert os.path.exists(
        state_dict_path
    ), f"No `pytorch_model.bin` file found in {model_dir}"
    loaded_state_dict = torch.load(state_dict_path)
    state_keys = loaded_state_dict.keys()
    keys_to_remove = [
        k for k in state_keys if k.startswith("regression") or k.startswith("norm")
    ]

    new_state_dict = OrderedDict({**loaded_state_dict})
    for k in keys_to_remove:
        del new_state_dict[k]
    return new_state_dict
###mode
for mode in ['test','valid','train']:


    #####model

    config = RobertaConfig.from_pretrained(
        'models_pre/ChemBERTa-77M-MLM', use_auth_token=True
    )
    state_dict = prune_state_dict('models_pre/ChemBERTa-77M-MLM')
    mol_encoder = RobertaForSequenceClassification.from_pretrained(
        'models_pre/ChemBERTa-77M-MLM',
        config=config,
        state_dict=state_dict,
        use_auth_token=True,
    )
    mol_encoder = mol_encoder.cuda()
    ####data
    mol_tokenizer = AutoTokenizer.from_pretrained('models_pre/ChemBERTa-77M-MLM', max_len=512, use_auth_token=True)
    vocab = [token.strip().split('_') for token in open('/home/ywh/flourine_smile_68test/data/data18/'+mode+'_set')]
    dict = {}
    for i,line in enumerate(vocab):
        f = mol_tokenizer([line[0]], truncation=True, padding=True)['input_ids'][0]
        nf = mol_tokenizer([line[2]], truncation=True, padding=True)['input_ids'][0]
        f = torch.tensor(f, dtype=torch.long).unsqueeze(0).cuda()
        nf = torch.tensor(nf, dtype=torch.long).unsqueeze(0).cuda()
        f_emb = mol_encoder(f)[0]
        nf_emb = mol_encoder(nf)[0]
        f_out = f_emb.squeeze(0).cpu().detach().numpy()
        nf_out = nf_emb.squeeze(0).cpu().detach().numpy()
        dict[str(i)+'_f']=f_out
        dict[str(i) + '_nf'] = nf_out
    np.savez('/home/ywh/flourine_smile_68test/data/data18/'+mode+'_full_emb.npz',**dict)
###

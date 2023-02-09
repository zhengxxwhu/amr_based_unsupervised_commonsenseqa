from tqdm import tqdm
import random
from transformers.models.roberta.modeling_roberta import RobertaConfig
from transformers.models.albert.modeling_albert import AlbertConfig
from modeling import RobertaForMaskedLM,AlbertForMaskedLM
from transformers.models.roberta.tokenization_roberta import RobertaTokenizer
from transformers.models.albert.tokenization_albert import AlbertTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import Dataset,DataLoader
import torch
import numpy as np
import spacy
from nltk.corpus import stopwords


skip_words = set(stopwords.words('english'))
skip_words.add('\'s')
skip_words.add('.')
skip_words.add(',')
nlp=spacy.load("en_core_web_sm")


class MLMDataset(Dataset):

	def __init__(self, data, pad_token, mask_token, max_words_to_mask, is_eval_test=False):
		self.data = data
		if not is_eval_test:
			random.shuffle(self.data)
		self.pad_token = pad_token
		self.mask_token = mask_token
		self.max_words_to_mask = max_words_to_mask

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		sample = self.data[idx]
		# sample=[sequences, label_ids, answer_key]
		return sample, self.pad_token, self.mask_token, self.max_words_to_mask


def mlm_CollateFn(batch,is_evaluate=False):
	batch_input_ids = []
	batch_input_mask = []
	batch_input_labels = []
	batch_cls_input_ids = []
	batch_cls_input_mask = []
	batch_label_ids = []
	#features=[sequences, label_ids, answer_key]
	features = [b[0] for b in batch]
	pad_token = batch[0][1]
	mask_token = batch[0][2]
	MAX_WORDS_TO_MASK = batch[0][3]
	#max_len = max([len(cand) for f in features for cand in f[0]])
	max_len_q = max([len(cand) for f in features for cand in f[0]])
	max_len=max([len(sq) for f in features for cand in f[0] for sq in cand])
	for f in features:
		batch_input_ids.append([])
		batch_input_mask.append([])
		batch_input_labels.append([])
		batch_cls_input_ids.append([])
		batch_cls_input_mask.append([])
		batch_label_ids.append(f[2])
		#f[0] sequences
		#for i in range(len(f[0])):
		for i in range(len(f[0])):
			multi_q_masked_sequences=[]
			multi_q_this_att_mask=[]
			multi_q_masked_labels = []
			multi_q_cls_sequences=[]
			multi_q_cls_att_mask = []
			for q in range(len(f[0][i])):
				masked_sequences = []
				masked_labels = []
				this_att_mask = []
				sequence = f[0][i][q] + [pad_token] * (max_len - len(f[0][i][q]))
				label_sequence = f[1][i][q] + [-100] * (max_len - len(f[1][i][q]))
				valid_indices = [l_i for l_i, l in enumerate(label_sequence) if l != -100]
				if len(valid_indices) > MAX_WORDS_TO_MASK and not is_evaluate:
					rm_indices = random.sample(valid_indices, (len(valid_indices) - MAX_WORDS_TO_MASK))
					label_sequence = [-100 if l_i in rm_indices else l for l_i, l in enumerate(label_sequence)]
				for j, t in enumerate(label_sequence):
					if t == -100:
						continue
					else:
						masked_sequences.append(sequence[:j] + [mask_token] + sequence[j + 1:])
						masked_labels.append([-100] * j + [sequence[j]] + [-100] * (max_len - j - 1))
					this_att_mask.append([1] * len(f[0][i][q]) + [0] * (max_len - len(f[0][i][q])))
				#理论上q为0即原文局不应当没有有意义的词在问句中
				if q!=0 and len(valid_indices)==0:
					print("sub_Q no meaning")
					continue
				#新增
				elif q==0 and len(valid_indices)==0:
					multi_q_masked_sequences = []
					multi_q_this_att_mask = []
					multi_q_masked_labels = []
					multi_q_cls_sequences = []
					multi_q_cls_att_mask = []
					break
				#
				multi_q_masked_sequences.append(torch.tensor(masked_sequences, dtype=torch.long))
				multi_q_this_att_mask.append(torch.tensor(this_att_mask, dtype=torch.long))
				multi_q_masked_labels.append(torch.tensor(masked_labels, dtype=torch.long))
				multi_q_cls_sequences.append(torch.tensor(sequence, dtype=torch.long))
				#multi_q_cls_att_mask.append(torch.tensor(this_att_mask[0], dtype=torch.long))
				multi_q_cls_att_mask.append(torch.tensor([1] * len(f[0][i][q]) + [0] * (max_len - len(f[0][i][q])), dtype=torch.long))
			#multi_q_masked_sequences=multi_q_masked_sequences+(max_len_q-len(multi_q_masked_sequences))*([pad_token]*max_len)
			#新增
			if len(multi_q_masked_sequences)>0:
			#
				batch_input_ids[-1].append(multi_q_masked_sequences)
				batch_input_mask[-1].append(multi_q_this_att_mask)
				batch_input_labels[-1].append(multi_q_masked_labels)
				batch_cls_input_ids[-1].append(multi_q_cls_sequences)
				batch_cls_input_mask[-1].append(multi_q_cls_att_mask)

	return batch_input_ids, batch_input_mask, batch_input_labels, batch_cls_input_ids, batch_cls_input_mask, torch.tensor(batch_label_ids, dtype=torch.long)


def mlm_step(batch_input_ids, batch_input_mask, batch_input_labels, args, model, CE, is_evaluate=False):
	num_cand = len(batch_input_ids[0])
	choice_loss = []
	choice_seq_lens = np.array([0] + [len(q) for sample in batch_input_ids for c in sample for q in c])
	choice_seq_lens = np.cumsum(choice_seq_lens)
	input_ids = torch.cat([q for sample in batch_input_ids for c in sample for q in c], dim=0).to(args.device)
	att_mask = torch.cat([q for sample in batch_input_mask for c in sample for q in c], dim=0).to(args.device)
	input_labels = torch.cat([q for sample in batch_input_labels for c in sample for q in c], dim=0).to(args.device)

	if len(input_ids) < args.max_sequence_per_time or is_evaluate:
		inputs = {'input_ids': input_ids,
				  'attention_mask': att_mask}
		outputs = model(**inputs)
		ce_loss = CE(outputs[0].view(-1, outputs[0].size(-1)), input_labels.view(-1))
		ce_loss = ce_loss.view(outputs[0].size(0), -1).sum(1)
	else:
		ce_loss = []
		for chunk in range(0, len(input_ids), args.max_sequence_per_time):
			inputs = {'input_ids': input_ids[chunk:chunk + args.max_sequence_per_time],
					  'attention_mask': att_mask[chunk:chunk + args.max_sequence_per_time]}
			outputs = model(**inputs)
			tmp_ce_loss = CE(outputs[0].view(-1, outputs[0].size(-1)),
							 input_labels[chunk:chunk + args.max_sequence_per_time].view(-1))
			tmp_ce_loss = tmp_ce_loss.view(outputs[0].size(0), -1).sum(1)
			ce_loss.append(tmp_ce_loss)
		ce_loss = torch.cat(ce_loss, dim=0)
	# all tokens are valid
	# 计算masked的平均Loss值
	for c_i in range(len(choice_seq_lens) - 1):
		start = choice_seq_lens[c_i]
		end = choice_seq_lens[c_i + 1]
		choice_loss.append(-ce_loss[start:end].sum() / (end - start))

	choice_loss = torch.stack(choice_loss)
	batch_choice_loss=[]

	batch_choice_seq_lens = np.array([0] + [sum([len(c) for c in sample]) for sample in batch_input_ids])
	batch_choice_seq_lens = np.cumsum(batch_choice_seq_lens)

	for b_i in range(len(batch_choice_seq_lens) - 1):
		start = batch_choice_seq_lens[b_i]
		end = batch_choice_seq_lens[b_i + 1]
		batch_choice_loss.append(choice_loss[start:end].view(num_cand,-1))

	return batch_choice_loss


def cls_step(batch_input_ids,batch_input_mask,args,model):
	num_cand = len(batch_input_ids[0])
	input_ids = torch.cat([q.unsqueeze(0) for sample in batch_input_ids for c in sample for q in c], dim=0).to(args.device)
	att_mask = torch.cat([q.unsqueeze(0) for sample in batch_input_mask for c in sample for q in c], dim=0).to(args.device)
	inputs = {'input_ids': input_ids,
			  'attention_mask': att_mask}
	outputs = model(**inputs)
	cls_embed=outputs[1][:,0,:]

	batch_cls_embed = []
	batch_choice_seq_lens = np.array([0] + [sum([len(c) for c in sample]) for sample in batch_input_ids])
	batch_choice_seq_lens = np.cumsum(batch_choice_seq_lens)
	for b_i in range(len(batch_choice_seq_lens) - 1):
		start = batch_choice_seq_lens[b_i]
		end = batch_choice_seq_lens[b_i + 1]
		batch_cls_embed.append(cls_embed[start:end,:].view(num_cand,-1,cls_embed.shape[-1]))

	return batch_cls_embed


def mlm_evaluate_dataset_acc(args, model, cls_classifier, embedder, eval_dataset, is_eval_test=False):
	if not is_eval_test:
		eval_dataloader = DataLoader(eval_dataset, shuffle=True, batch_size=args.eval_batch_size,
									 #collate_fn=lambda x:mlm_CollateFn(x,is_evaluate))
									 collate_fn=lambda x: mlm_CollateFn(x, is_evaluate=True))
	else:
		eval_dataloader = DataLoader(eval_dataset, shuffle=False, batch_size=args.eval_batch_size,
									 # collate_fn=lambda x:mlm_CollateFn(x,is_evaluate))
									 collate_fn=lambda x: mlm_CollateFn(x, is_evaluate=True))

	print("***** Running evaluation *****")
	CE = torch.nn.CrossEntropyLoss(reduction='none')
	preds = []
	out_label_ids = []
	#暂时
	#for batch in tqdm(eval_dataloader):
	model.eval()
	with torch.no_grad():
		for batch in tqdm(eval_dataloader):
			batch_input_ids, batch_input_mask, batch_input_labels, batch_cls_input_ids, batch_cls_input_mask, batch_label_ids = batch
			if args.with_sub_Q:
				if args.fix_PLM:
					num_cand = len(batch_cls_input_ids[0])
					input_ids = torch.cat([q.unsqueeze(0) for sample in batch_cls_input_ids for c in sample for q in c],
										  dim=0).to(args.device)
					att_mask = torch.cat([q.unsqueeze(0) for sample in batch_cls_input_mask for c in sample for q in c],
										 dim=0).to(args.device)
					inputs = {'input_ids': input_ids,
							  'attention_mask': att_mask}
					outputs = embedder(**inputs)
					#cls_embed = outputs[1]
					cls_embed = outputs[:,0,:]

					batch_cls_embed = []
					# [batch_size,candidate_num*question_num]
					batch_choice_seq_lens = np.array(
						[0] + [sum([len(c) for c in sample]) for sample in batch_cls_input_ids])
					batch_choice_seq_lens = np.cumsum(batch_choice_seq_lens)
					for b_i in range(len(batch_choice_seq_lens) - 1):
						start = batch_choice_seq_lens[b_i]
						end = batch_choice_seq_lens[b_i + 1]
						# [batch_size,(candidate_num,question_num,embedding_size)]
						batch_cls_embed.append(cls_embed[start:end, :].view(num_cand, -1, cls_embed.shape[-1]))
					# batch_cls_embed = cls_step(batch_cls_input_ids, batch_cls_input_mask, args, embedder)

					torch.no_grad()
				else:
					# batch_cls_embed [batch_size,(candidate_num,question_num,embedding_size)]
					batch_cls_embed = cls_step(batch_cls_input_ids, batch_cls_input_mask, args, model)
				# batch_choice_loss [batch_size,(candidate_num,question_num)]
				batch_choice_loss = mlm_step(batch_input_ids, batch_input_mask, batch_input_labels, args, model, CE)

				if args.fix_PLM:
					torch.enable_grad()

				batch_weighted_sum_l = []
				for cls_embed, choice_loss in zip(batch_cls_embed, batch_choice_loss):
					# (cand_num)
					weighted_sum_l = cls_classifier(cls_embed, choice_loss)
					batch_weighted_sum_l.append(weighted_sum_l.unsqueeze(0))
				# (batch_size,cand_num)
				batch_weighted_sum_l = torch.cat(batch_weighted_sum_l, dim=0)

				preds.append(batch_weighted_sum_l)
			else:
				single_batch_input_ids = [[[c[0]] for c in sample] for sample in batch_input_ids]
				single_batch_input_mask = [[[c[0]] for c in sample] for sample in batch_input_mask]
				single_batch_input_labels = [[[c[0]] for c in sample] for sample in batch_input_labels]
				# batch_choice_loss [batch_size,(candidate_num,1)]
				single_batch_choice_loss = mlm_step(single_batch_input_ids, single_batch_input_mask,
													single_batch_input_labels, args, model, CE)
				# (batch_size,cand_num)
				batch_choice_loss = torch.cat(
					[choice_loss.squeeze(-1).unsqueeze(0) for choice_loss in single_batch_choice_loss])
				preds.append(batch_choice_loss)

			out_label_ids.append(batch_label_ids.numpy())
		preds = torch.cat(preds, dim=0).cpu().numpy()
		preds = np.argmax(preds, axis=1)

		if is_eval_test:
			return preds

		eval_acc=(preds == np.concatenate(out_label_ids, axis=0)).mean()
		print('eval acc:{}'.format(eval_acc))

	return eval_acc


def load_loss_fn(args):
	loss_fns={
		'mlm_CE': torch.nn.CrossEntropyLoss(),
		'mlm_margin': torch.nn.MultiMarginLoss(margin=args.margin),
		'binary_CE': torch.nn.BCELoss()
	}
	return loss_fns[args.loss_type]


from transformers import MODEL_WITH_LM_HEAD_MAPPING
MODEL_CONFIG_CLASSES = list(MODEL_WITH_LM_HEAD_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)
MODEL_CLASSES = {
    'roberta-mlm': (RobertaConfig, RobertaForMaskedLM, RobertaTokenizer),
	'albert-mlm': (AlbertConfig,AlbertForMaskedLM , AlbertTokenizer)
}

def load_config_tokenizer_model(args):
	args.model_type = args.model_type.lower()
	config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
	config = config_class.from_pretrained(args.model_name_or_path)
	tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
	model = model_class.from_pretrained(args.model_name_or_path, config=config)
	return config,tokenizer,model


def count_parameters(model):
	return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_optimizer_scheduler(args, dataloader_length, *modules_lrs):
	t_total = dataloader_length // args.gradient_accumulation_steps * args.num_train_epochs

	# Prepare optimizer and schedule (linear warmup and decay)
	no_decay = ['bias', 'LayerNorm.weight']
	#decay_params=[]
	#no_decay_params=[]
	optimizer_grouped_parameters=[]
	for module,lr in modules_lrs:
		if module==None:
			continue
		#decay_params.extend([p for n, p in module.named_parameters() if not any(nd in n for nd in no_decay)])
		#no_decay_params.extend([p for n, p in module.named_parameters() if any(nd in n for nd in no_decay)])
		decay_params=[p for n, p in module.named_parameters() if not any(nd in n for nd in no_decay)]
		no_decay_params=[p for n, p in module.named_parameters() if any(nd in n for nd in no_decay)]
		optimizer_grouped_parameters.append({
			'params':decay_params,
			'weight_decay':args.weight_decay,
			'lr':lr
		})
		optimizer_grouped_parameters.append({
			'params': no_decay_params,
			'weight_decay': 0.0,
			'lr':lr
		})

	warmup_steps = args.warmup_steps if args.warmup_steps != 0 else int(args.warmup_proportion * t_total)
	#optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon, betas=(0.9, 0.98))
	optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon, betas=(0.9, 0.98))
	scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)

	return optimizer, scheduler

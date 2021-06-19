from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from .cnn import get_cnn
import re

class RDTC(nn.Module):
    def __init__(self, num_classes,
                 dataset, decision_size=2, max_iters=30, attribute_size=20,
                 attribute_mtx=None, attribute_coef=0., hidden_size=100,
                 tau_initial=5, tau_target=0.5, use_pretrained=False,
                 threshold=1.):
        super().__init__()
        self.num_classes = num_classes          # 200
        self.attribute_size = attribute_size    # 312
        self.attribute_mtx = attribute_mtx      # torch.Size([200, 312])
        self.attribute_coef = attribute_coef if attribute_mtx is not None else 0.   # 0.2
        self.decision_size = decision_size      # 2
        self.tau_initial = tau_initial          # 5
        self.tau_target = tau_target            # 0.5
        self.max_iters = max_iters              # 25
        self.threshold = threshold              # 1

        self.stats = defaultdict(list)

        assert decision_size == 2 or (decision_size > 2 and self.attribute_coef == 0.), \
            'Attribute loss only supported for decision_size == 2'

        self.cnn, cnn_out_size = self.init_cnn(dataset, use_pretrained)   
        # cnn_out_size: 2048

        self.init_network(hidden_size, decision_size, num_classes,
                          attribute_size, cnn_out_size)

        self.init_losses()

    def init_network(self, hidden_size, decision_size, num_classes,
                     attribute_size, cnn_out_size):                     
       	# Parameters:
       	#	 hidden_size: 1024

        assert decision_size > 1

        # LSTM initialization parameters
        self.init_h0 = nn.Parameter(
            torch.zeros(hidden_size).uniform_(-0.01, 0.01), requires_grad=True
        )
        self.init_c0 = nn.Parameter(
            torch.zeros(hidden_size).uniform_(-0.01, 0.01), requires_grad=True
        )

        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)     
        # LSTM(1024, 1024, batch_first=True)

        self.classifier = nn.Sequential(
                nn.Linear(attribute_size * decision_size, hidden_size),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(hidden_size),
                nn.Linear(hidden_size, num_classes)
        )

        # self.classifier

        # Sequential(
        #   (0): Linear(in_features=624, out_features=1024, bias=True)
        #   (1): ReLU(inplace=True)
        #   (2): BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #   (3): Linear(in_features=1024, out_features=200, bias=True)
        # )

        self.question_mlp = nn.Sequential(
                nn.BatchNorm1d(hidden_size),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(hidden_size),
                nn.Linear(hidden_size, attribute_size)
        )

        # self.question_mlp

        # Sequential(
        #   (0): BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #   (1): Linear(in_features=1024, out_features=1024, bias=True)
        #   (2): ReLU(inplace=True)
        #   (3): BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #   (4): Linear(in_features=1024, out_features=312, bias=True)
        # )

        self.attribute_mlp = nn.Sequential(
                nn.BatchNorm1d(cnn_out_size),
                nn.Linear(cnn_out_size, hidden_size),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(hidden_size),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(hidden_size),
                nn.Linear(hidden_size, attribute_size * decision_size)
        )

        # self.attribute_mlp

        # Sequential(
        #   (0): BatchNorm1d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #   (1): Linear(in_features=2048, out_features=1024, bias=True)
        #   (2): ReLU(inplace=True)
        #   (3): BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #   (4): Linear(in_features=1024, out_features=1024, bias=True)
        #   (5): ReLU(inplace=True)
        #   (6): BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #   (7): Linear(in_features=1024, out_features=624, bias=True)
        # )

        self.pre_lstm = nn.Sequential(
                nn.Linear(2 * attribute_size * decision_size, hidden_size),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(hidden_size)
        )

        # self.pre_lstm

        # Sequential(
        #   (0): Linear(in_features=1248, out_features=1024, bias=True)
        #   (1): ReLU(inplace=True)
        #   (2): BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # )
        

        # Temperature parameters
        self.attribute_mlp.tau = nn.Parameter(torch.tensor([self.tau_initial],
                                              dtype=torch.float), requires_grad=True)   # Parameter containing: tensor([5.], requires_grad=True)
        self.question_mlp.tau = nn.Parameter(torch.tensor([self.tau_initial],
                                             dtype=torch.float), requires_grad=True)    # Parameter containing: tensor([5.], requires_grad=True)


    def init_losses(self):
        self.cls_loss = nn.CrossEntropyLoss()
        self.attr_loss = nn.BCEWithLogitsLoss()

    def init_cnn(self, dataset, use_pretrained):
        if use_pretrained:
            cnn_state_dict = torch.load('pretrained/{}_resnet152.pkl'.format(dataset))  
            # cnn_state_dict: 777
            cnn, cnn_out_size = get_cnn(cnn_state_dict, freeze_weights=True)
        else:
            cnn, cnn_out_size = get_cnn()

        return cnn, cnn_out_size

    def get_initial_state(self, batch_size):
        h0 = self.init_h0.view(1, 1, -1).expand(-1, batch_size, -1) # torch.Size([1, 128, 1024])
        c0 = self.init_c0.view(1, 1, -1).expand(-1, batch_size, -1) # torch.Size([1, 128, 1024])
        state = (h0.contiguous(), c0.contiguous())
        return state

    def argmax(self, y_soft, dim):
        # Differentiable argmax
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(y_soft).scatter_(dim, index, 1.0)
        argmax = y_hard - y_soft.detach() + y_soft
        return argmax

    def reset_stats(self):
        self.unique_attributes = [set() for i in range(self.max_iters)]
        if self.attribute_coef > 0.:
            self.attr_pred_correct = np.zeros(self.max_iters)
        self.n_pruned = np.zeros(self.max_iters)

    def update_unique_attributes(self, attribute_idx):
        for iter, attr_idx in enumerate(attribute_idx):
            unique_attributes = attr_idx.unique().cpu().numpy()
            self.unique_attributes[iter] = self.unique_attributes[iter].union(unique_attributes)

    def get_unique_attributes(self):
        uniq_per_iter = []
        for i in range(self.max_iters):
            iter_set = self.unique_attributes[i]
            for j in range(i+1):
                if j == i:
                    continue
                iter_set = iter_set.union(self.unique_attributes[j])
            uniq_per_iter.append(len(iter_set))
        return uniq_per_iter

    def update_attr_preds(self, attr_correct):
        self.attr_pred_correct += attr_correct.cpu().numpy()

    def get_attr_acc(self, total_cnt):
        correct_cumsum = np.cumsum(self.attr_pred_correct)
        cnt_per_iter = (np.arange(self.max_iters) + 1) * total_cnt
        return correct_cumsum / cnt_per_iter

    def update_pruning_stats(self, threshold_masks):
        n_pruned = torch.stack(threshold_masks).sum(1).cpu().numpy()
        self.n_pruned += n_pruned

    def get_pruning_ratio(self, total_cnt):
        return self.n_pruned / total_cnt

    def apply_threshold(self, classification, threshold_mask, threshold_classification):

        above_thres = (F.softmax(classification, dim=1).max(dim=1)[0] > self.threshold)
        new_thres = (above_thres.int() - threshold_mask.int()).clamp(0., 1.).bool()
        threshold_classification[new_thres] = classification[new_thres]
        threshold_mask = threshold_mask | above_thres
        classification[threshold_mask] = threshold_classification[threshold_mask]

        return classification, threshold_mask, threshold_classification

    def compute_loss(self, labels, classifications, attribute_idx,
                     bin_attribute_logits=None):
    	# Paramters:
    	#	labels: 128
    	#	classifications: 25 (torch.Size([128, 200]))
    	#	attribute_idx: 25 (128)
    	#	bin_attribute_logits: torch.Size([128, 312])

        # Update attribute stats
        self.update_unique_attributes(attribute_idx)

        # Prepare dimensions
        n_iter = len(classifications)
        # n_iter: 25

        iter_labels = labels.repeat(n_iter)
        # iter_labels: torch.Size([3200]) (128*25)

        classifications = torch.cat(classifications, dim=0)
        #  classifications: torch.Size([3200, 200])

        attribute_idx = torch.cat(attribute_idx, dim=0)
        # attribute_idx: torch.Size([3200])

        if bin_attribute_logits is not None:
            bin_attribute_logits = bin_attribute_logits.repeat(n_iter, 1)
            # bin_attribute_logits: torch.Size([3200, 312])

        # RDT loss
        loss = (1. - self.attribute_coef) * self.cls_loss(classifications, iter_labels)

        # Attribute loss
        if self.attribute_coef > 0.:
            attribute_target = self.attribute_mtx[iter_labels, :].gather(1, attribute_idx.unsqueeze(1)).squeeze()
            # attribute_target: torch.Size([3200])
            attribute_pred = bin_attribute_logits.gather(1, attribute_idx.unsqueeze(1)).squeeze()
            # attribute_pred: torch.Size([3200])

            loss += self.attribute_coef * self.attr_loss(attribute_pred,
                                                         attribute_target)

            # Update running attribute prediction accuracy
            attribute_pred_bin = (attribute_pred > 0.).long()
            #  attribute_pred_bin: torch.Size([3200])

            self.update_attr_preds((attribute_pred_bin == attribute_target).view(n_iter, -1).sum(1))

        return loss

    def attribute_based_learner(self, images):
    	# Parameters:
    	#	images: images: torch.Size([128, 3, 224, 224])

        img_feats = self.cnn(images) 
        # img_feats: torch.Size([128, 2048])

        img_feats = img_feats.view(img_feats.size(0), -1)
        # img_feats: torch.Size([128, 2048])

        image_features = self.attribute_mlp(img_feats) 
        # image_features: torch.Size([128, 624])

        attribute_logits = image_features.view(-1, self.decision_size) 
        # attribute_logits:  torch.Size([39936, 2])

        attributes_softmax = F.softmax(attribute_logits / self.attribute_mlp.tau, dim=1) 
        # attributes_softmax: torch.Size([39936, 2])

        attributes_hard = self.argmax(attributes_softmax, dim=1)    
        # attributes_hard: torch.Size([39936, 2])

        image_features = attributes_hard.view(images.size(0), -1, self.decision_size) 
        # image_features: torch.Size([128, 312, 2])

        if self.attribute_coef > 0.:
            # Binary logits for attribute loss
            bin_attribute_logits = attribute_logits - attribute_logits[:, 1].unsqueeze(-1)
            bin_attribute_logits = bin_attribute_logits[:, 0].view(images.size(0), -1)
            # bin_attribute_logits: torch.Size([128, 312])
        else:
            bin_attribute_logits = None

        return image_features, bin_attribute_logits

    def make_decision(self, lstm_out, binary_features):
        # Parameters:  
        #	lstm_out: torch.Size([128, 1024])
        #	binary_features: torch.Size([128, 312, 2])

        # Perform categorical feature selection
        selection_logits = self.question_mlp(lstm_out) 
        # selection_logits: torch.Size([128, 312])

        if self.training:
            hard_selection = F.gumbel_softmax(selection_logits, hard=True,
                                              tau=self.question_mlp.tau)     
            # hard_selection: torch.Size([128, 312])

        else:
            hard_selection = self.argmax(selection_logits, dim=1)
            # hard_selection: torch.Size([128, 312])

        # Get single decision

        decision = (hard_selection.unsqueeze(2) * binary_features) 
        # decision: torch.Size([128, 312, 2])

        # Index of decision
        attribute_idx = hard_selection.max(dim=1)[1]
        # attribute_idx: 128
        
        '''Added code'''

        image_id = 0
        #print(attribute_idx[image_id])
        decision_array = decision[image_id,:,:].cpu().detach().numpy()

        decision_index = np.argmax(decision_array.sum(1))

        decision_for_attribute = decision[image_id,decision_index,:].cpu().detach().numpy()

        decision_value = 0
        if decision_for_attribute[0] == 1:
        	decision_value = 1

        f = open("data/cub/attributes.txt", "r")
        for i, line in enumerate(f):
        	if(i==attribute_idx[image_id]-1):
        		attribute_name = line[3:]
        print("Attribute_name:{}, decision:{}\n".format(attribute_name,decision_value))
        f.close()

        decision = decision.view(-1, self.attribute_size * self.decision_size)   
        # decision: torch.Size([128, 624])

        return decision, attribute_idx

    def run_rdt_iteration(self, binary_features, state, explicit_memory):
    	# Parameters:
    	#	binary_features: torch.Size([128, 312, 2])
    	#   state: (torch.Size([1, 128, 1024]), torch.Size([1, 128, 1024]))
    	#	explicit_memory: None

        lstm_out = state[0].squeeze(0)   
        # lstm_out: torch.Size([128, 1024])

        # Make binary decision

        decision, attribute_idx = self.make_decision(lstm_out, binary_features)
        # decision: torch.Size([128, 624])
        # attribute_idx: 128

        if explicit_memory is None:
            explicit_memory = decision
            # explicit_memory: torch.Size([128, 624])
        else:
            explicit_memory = (explicit_memory + decision).clamp(0., 1.)

        # Apply scaling similar to dropout scaling
        scaled_em = explicit_memory / explicit_memory.sum(dim=1).unsqueeze(1).detach()  
        # scaled_em: torch.Size([128, 624])

        lstm_in = torch.cat((scaled_em, decision), dim=1)   
        # lstm_in: torch.Size([128, 1248])

        # Update LSTM state
        lstm_in = self.pre_lstm(lstm_in).unsqueeze(1) 
        # lstm_in: torch.Size([128, 1, 1024])

        _, state = self.lstm(lstm_in, state) 
        # state: (torch.Size([1, 128, 1024]), torch.Size([1, 128, 1024]))

        # Get current classification
        classification = self.classifier(scaled_em)  
        # classification: torch.Size([128, 200])
        #print(classification[0,:])

        return classification, state, explicit_memory, attribute_idx

    def recurrent_decision_tree(self, binary_features, labels): 
    	# Outputs for every iteration

    	# Parameters: 
    	#	binary_features: torch.Size([128, 312, 2])
    	# 	labels: torch.Size([128])

        all_classifications = []
        all_attribute_idx = []

        if not self.training and self.threshold < 1.:   
        # self.training: True
        # self.threshold: 1.0
            all_threshold_masks = []

            threshold_mask = torch.zeros(len(labels), dtype=torch.bool).to(labels.device)
    
            threshold_classification = torch.zeros(labels.size(0), self.num_classes).to(labels.device)
        else:
            all_threshold_masks = None

        # Set initial state
        state = self.get_initial_state(binary_features.size(0))  
        # state: (torch.Size([1, 128, 1024]), torch.Size([1, 128, 1024]))

        explicit_memory = None

        for j in range(self.max_iters):
            
            classification, state, explicit_memory, attribute_idx = self.run_rdt_iteration(  
                binary_features, state, explicit_memory
            )
            
            # classification: torch.Size([128, 200])
            # state: (torch.Size([1, 128, 1024]), torch.Size([1, 128, 1024]))
            # explicit_memory: torch.Size([128, 624])
            # attribute_idx: 128

            if not self.training and self.threshold < 1.:
                all_threshold_masks.append(threshold_mask.clone())

                classification, threshold_mask, threshold_classification = self.apply_threshold(
                    classification, threshold_mask, threshold_classification
                )

            image_id = 0
            classification_image = classification[image_id,:].cpu().detach().numpy()
            #predicted_class_id = np.argmax(classification_image)

            top_prediciton_ids = classification_image.argsort()[-3:][::-1]
            top_predictions_values = classification_image[top_prediciton_ids]
            normalized_top_predictions_values = top_predictions_values/np.sum(top_predictions_values)

            class_names = []
            f1 = open("data/cub/classes.txt", "r")
            for i, line in enumerate(f1):
            	class_name = line[3:-1]
            	pattern = r'[0-9]'
            	class_name = re.sub(pattern, '', class_name)
            	class_name = class_name.replace('.','')
            	class_names.append(class_name) 

            	#if(i==predicted_class_id):
            		#predicted_class = line[3:]

            top_classes = [class_names[k] for k in top_prediciton_ids]
            print(top_classes, normalized_top_predictions_values)		
            #print("Predicted class: {}\n".format(predicted_class))
            all_classifications.append(classification)
            all_attribute_idx.append(attribute_idx)

        return all_classifications, all_attribute_idx, all_threshold_masks # 25, 25, None

    # def tensor_to_PIL(tensor): 
    # 	unloader = transforms.ToPILImage()
    # 	image = tensor.cpu().clone() 
    # 	image = image.squeeze(0) 
    # 	image = unloader(image) 
    # 	return image
    
    def forward(self, images, labels):
        # Get categorical features once

        # Parameters:
        #	images: torch.Size([128, 3, 224, 224]) 
        #	labels: torch.Size([128])

        tensor_image = images[0,:,:,:]
        tensor_image = tensor_image.permute(1, 2, 0)
        tensor_image = tensor_image.cpu().detach().numpy()

        # tensor_label = labels[0].cpu().detach().numpy()
        # print(tensor_label)

        # plt.imshow(tensor_image)
        # plt.show()

        # binary_features(attributes or features)?
        binary_features, bin_attribute_logits = self.attribute_based_learner(images)    
        # binary_features: torch.Size([128, 312, 2])
        # bin_attribute_logits: torch.Size([128, 312])

        classification, attribute_idx, thres_mask = self.recurrent_decision_tree(      
                binary_features, labels
        )
        # classification: 25
        # attribute_idx: 25
        # thres_mask: None (for evaluation)

        if thres_mask is not None:
            self.update_pruning_stats(thres_mask)

        loss = self.compute_loss(labels, classification, attribute_idx,
                                 bin_attribute_logits)

        return classification, loss

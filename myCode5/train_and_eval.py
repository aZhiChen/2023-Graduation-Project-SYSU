import numpy as np
import copy
import torch
import dgl
import time
from utils import set_seed
from torch import nn
"""
1. Train and eval
"""


def train(model, data, feats, vectors, labels, criterion, optimizer, idx_train, lamb=1):
    """
    GNN full-batch training. Input the entire graph `g` as data.
    lamb: weight parameter lambda
    """
    model.train()

    # Compute loss and prediction
    h_list, logits = model(data, feats, vectors)
    out = logits.log_softmax(dim=1)
    loss = criterion(out[idx_train], labels[idx_train])
    loss_val = loss.item()

    loss *= lamb
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return h_list, loss_val


def train_sage(model, dataloader, feats, vectors, labels, criterion, optimizer, lamb=1):
    """
    Train for GraphSAGE. Process the graph in mini-batches using `dataloader` instead the entire graph `g`.
    lamb: weight parameter lambda
    """
    device = feats.device
    model.train()
    total_loss = 0
    for step, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
        blocks = [blk.int().to(device) for blk in blocks]
        batch_feats = feats[input_nodes]
        batch_labels = labels[output_nodes]

        # Compute loss and prediction
        h_list, logits = model(blocks, batch_feats, vectors, input_nodes)
        out = logits.log_softmax(dim=1)
        loss = criterion(out, batch_labels)
        total_loss += loss.item()

        loss *= lamb
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return h_list, total_loss / len(dataloader)


#0326版本

def train_mini_batch(model, feats, vectors, labels, batch_size, criterion, optimizer, lamb=1,  distill = None, hiddens = None, Idx = None):
    """
    Train MLP for large datasets. Process the data in mini-batches. The graph is ignored, node features only.
    lamb: weight parameter lambda
    """
    model.train()
    #num_batches = 1

    #mini_batch
    #num_batches = max(1, feats.shape[0] // batch_size)
    #idx_batch = torch.randperm(feats.shape[0])[: num_batches * batch_size]
    

    ##full_batch
    num_batches = 1
    idx_batch = torch.randperm(feats.shape[0])[:]
    ####
    if num_batches == 1:
        idx_batch = idx_batch.view(1, -1)
    else:
        idx_batch = idx_batch.view(num_batches, batch_size)

    total_loss = 0
    h_list = []
    criterion_h = torch.nn.MSELoss()
    for i in range(num_batches):
        # No graph needed for the forward function

        h, logits = model(None, feats[idx_batch[i]], vectors, idx_batch[i])
        
        out = logits.log_softmax(dim=1)
        if distill != "hidden" :
            loss = criterion(out, labels[idx_batch[i]])
        if distill == "hidden" :
            max_num = len(h[0]) #100 200 400 800 80
            p = 0.2
            max_num = int(p * max_num)
            #print(max_num)
            idx = Idx[:max_num]
            #idx = Idx[:]
            #print(len(Idx))
            #print(len(hiddens))
            '''
            W_r = torch.eye(len(h[0])) * 1
            h_m = torch.matmul(h[0], h[0].T)
            h_m = torch.matmul(h_m, W_r)
            ht_m = torch.matmul(hiddens, hiddens.T)
            '''
            W_r = torch.eye(len(idx)) * 1
            h_m = torch.matmul(h[0][idx], h[0][idx].T)
            h_m = torch.matmul(h_m, W_r)
            ht_m = torch.matmul(hiddens[idx], hiddens[idx].T)
            #print("-----------------------------------h_m---------------------------------", h_m)
            #print("-----------------------------------ht_m--------------------------------", ht_m)
            
            loss_h = torch.norm( (h_m-ht_m) , p='fro') ** 2
            #print("loss_h: ",loss_h)
            #print("loss: ", loss)
            # loss_h = criterion_h(h[len(h)-1], hiddens[idx_batch[i]])
            
            loss = loss_h 
            #print("loss_h: ",loss)
            
        total_loss += loss.item()
        loss *= lamb
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        h_list = h_list + h
    
    return h_list, total_loss / num_batches


'''
#full_batch
#03.20版本
def train_mini_batch(model, feats, vectors, labels, batch_size, criterion, optimizer, lamb=1, alpha = 0, distill = None, hiddens = None, Idx = None):
    """
    Train MLP for large datasets. Process the data in mini-batches. The graph is ignored, node features only.
    lamb: weight parameter lambda
    """
    model.train()
    #num_batches = 1

    #mini_batch
    #num_batches = max(1, feats.shape[0] // batch_size)
    #idx_batch = torch.randperm(feats.shape[0])[: num_batches * batch_size]
    

    ##full_batch
    num_batches = 1
    idx_batch = torch.randperm(feats.shape[0])[:]
    ####
    if num_batches == 1:
        idx_batch = idx_batch.view(1, -1)
    else:
        idx_batch = idx_batch.view(num_batches, batch_size)

    total_loss = 0
    h_list = []
    criterion_h = torch.nn.MSELoss()
    for i in range(num_batches):
        # No graph needed for the forward function

        h, logits = model(None, feats[idx_batch[i]], vectors, idx_batch[i])
        
        out = logits.log_softmax(dim=1)
        loss = criterion(out, labels[idx_batch[i]])
        if distill == "hidden" :
            max_num = 120 #100 200 400 800 80
            idx = Idx[:]
            #idx = Idx[:]
            #print(len(Idx))
            #print(len(hiddens))
            W_r = torch.eye(len(idx)) * 1
            h_m = torch.matmul(h[0][idx], h[0][idx].T)
            h_m = torch.matmul(h_m, W_r)
            ht_m = torch.matmul(hiddens[idx], hiddens[idx].T)
            
            #print("-----------------------------------h_m---------------------------------", h_m)
            #print("-----------------------------------ht_m--------------------------------", ht_m)
            
            loss_h = torch.norm( (h_m-ht_m) , p='fro') ** 2
            #print("loss_h: ",loss_h * alpha)
            #print("loss: ", loss)
            # loss_h = criterion_h(h[len(h)-1], hiddens[idx_batch[i]])
            
            loss += loss_h * alpha
            
        total_loss += loss.item()
        
        loss *= lamb
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        h_list = h_list + h
    
    return h_list, total_loss / num_batches
'''
    



'''
    

def train_mini_batch(model, feats, vectors, labels, batch_size, criterion, optimizer, lamb=1, alpha = 0, distill = None, hiddens = None, Idx = None):
    """
    Train MLP for large datasets. Process the data in mini-batches. The graph is ignored, node features only.
    lamb: weight parameter lambda
    """
    model.train()
    #num_batches = max(1, feats.shape[0] // batch_size)

    num_batches = max(1, feats.shape[0] // batch_size)
    idx_batch = torch.randperm(feats.shape[0])[: num_batches * batch_size]


    if num_batches == 1:
        idx_batch = idx_batch.view(1, -1)
    else:
        idx_batch = idx_batch.view(num_batches, batch_size)

    total_loss = 0
    h_list = []
    criterion_h = torch.nn.MSELoss()
    for i in range(num_batches):
        # No graph needed for the forward function
        h, logits = model(None, feats[idx_batch[i]], vectors, idx_batch[i])
        #h_t = h.transpose(0,1)
        
        out = logits.log_softmax(dim=1)
        loss = criterion(out, labels[idx_batch[i]])
        #logits = model(None, feats[idx_batch[i]] )
        if distill != None:
            h_m = torch.matmul(h[len(h)-1], h[len(h)-1].T)
            ht_m = torch.matmul(hiddens[idx_batch[i]], hiddens[idx_batch[i]].T)
            loss_h = torch.norm( (h_m-ht_m) , p='fro') ** 2
            #print("loss_h: ",loss_h * 1e-8)
            #print("loss: ", loss)
            # loss_h = criterion_h(h[len(h)-1], hiddens[idx_batch[i]])
            
            loss += loss_h * alpha
            
        total_loss += loss.item()
        
        loss *= lamb
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        h_list = h_list + h
    
    return h_list, total_loss / num_batches

'''

def evaluate(model, data, feats, vectors, labels, criterion, evaluator, idx_eval=None):
    """
    Returns:
    out: log probability of all input data
    loss & score (float): evaluated loss & score, if idx_eval is not None, only loss & score on those idx.
    """
    model.eval()
    with torch.no_grad():
        h_list, logits = model.inference(data, feats, vectors)
        out = logits.log_softmax(dim=1)
        if idx_eval is None:
            loss = criterion(out, labels)
            score = evaluator(out, labels)
        else:
            loss = criterion(out[idx_eval], labels[idx_eval])
            score = evaluator(out[idx_eval], labels[idx_eval])
    return h_list, out, loss.item(), score


def evaluate_mini_batch(
    model, feats, vectors, labels, criterion, batch_size, evaluator, idx_eval=None
):
    """
    Evaluate MLP for large datasets. Process the data in mini-batches. The graph is ignored, node features only.
    Return:
    out: log probability of all input data
    loss & score (float): evaluated loss & score, if idx_eval is not None, only loss & score on those idx.
    """
    model.eval()
    with torch.no_grad():
        num_batches = int(np.ceil(len(feats) / batch_size))
        out_list = []
        h_list = []
        for i in range(num_batches):
            idx = []
            for j in range(batch_size * i, batch_size * (i + 1)):
                if j == len(feats):
                    break
                idx.append(j)
            idx = torch.tensor(idx)
            if batch_size * (i + 1) < len(feats):
                end = batch_size * (i + 1)
            else:
                end = len(feats)
            h, logits = model.inference(None, feats[batch_size * i : end ], vectors, idx )
            out = logits.log_softmax(dim=1) 
            out_list += [out.detach()]
            h_list = h_list + h
        out_all = torch.cat(out_list)

        if idx_eval is None:
            loss = criterion(out_all, labels)
            score = evaluator(out_all, labels)
        else:
            loss = criterion(out_all[idx_eval], labels[idx_eval])
            score = evaluator(out_all[idx_eval], labels[idx_eval])

    return h_list, out_all, loss.item(), score



"""
2. Run teacher
"""


def run_transductive(
    conf,
    model,
    g,
    feats,
    vectors,
    labels,
    indices,
    criterion,
    evaluator,
    optimizer,
    logger,
    loss_and_score,
):
    """
    Train and eval under the transductive setting.
    The train/valid/test split is specified by `indices`.
    The input graph is assumed to be large. Thus, SAGE is used for GNNs, mini-batch is used for MLPs.

    loss_and_score: Stores losses and scores.
    """
    set_seed(conf["seed"])
    device = conf["device"]
    batch_size = conf["batch_size"]

    idx_train, idx_val, idx_test = indices

    feats = feats.to(device)
    vectors = vectors.to(device)
    labels = labels.to(device)

    if "SAGE" in model.model_name:
        # Create dataloader for SAGE

        # Create csr/coo/csc formats before launching sampling processes
        # This avoids creating certain formats in each data loader process, which saves momory and CPU.
        g.create_formats_()
        sampler = dgl.dataloading.MultiLayerNeighborSampler(
            [eval(fanout) for fanout in conf["fan_out"].split(",")]
        )
        dataloader = dgl.dataloading.DataLoader(
            g,
            idx_train,
            sampler,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=conf["num_workers"],
        )

        # SAGE inference is implemented as layer by layer, so the full-neighbor sampler only collects one-hop neighors
        sampler_eval = dgl.dataloading.MultiLayerFullNeighborSampler(1)
        dataloader_eval = dgl.dataloading.DataLoader(
            g,
            torch.arange(g.num_nodes()),
            sampler_eval,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=conf["num_workers"],
        )

        data = dataloader
        data_eval = dataloader_eval
    elif "MLP" in model.model_name:
        feats_train, vectors_train,labels_train = feats[idx_train], vectors[idx_train], labels[idx_train]
        feats_val, vectors_val, labels_val = feats[idx_val], vectors[idx_val], labels[idx_val]
        feats_test, vectors_test, labels_test = feats[idx_test], vectors[idx_test], labels[idx_test]
    else:
        g = g.to(device)
        data = g
        data_eval = g

    best_epoch, best_score_val, count = 0, 0, 0
    for epoch in range(1, conf["max_epoch"] + 1):
        if "SAGE" in model.model_name:
            _, loss = train_sage(model, data, feats, vectors, labels, criterion, optimizer)
        elif "MLP" in model.model_name:
            _, loss = train_mini_batch(
                model, feats_train, vectors_train, labels_train, batch_size, criterion, optimizer
            )
        else:
            _, loss = train(model, data, feats, vectors, labels, criterion, optimizer, idx_train)

        if epoch % conf["eval_interval"] == 0:
            if "MLP" in model.model_name:
                _, _, loss_train, score_train = evaluate_mini_batch(
                    model, feats_train, vectors_train, labels_train, criterion, batch_size, evaluator
                )
                _, _, loss_val, score_val = evaluate_mini_batch(
                    model, feats_val, vectors_val, labels_val, criterion, batch_size, evaluator
                )
                _, _, loss_test, score_test = evaluate_mini_batch(
                    model, feats_test, vectors_test, labels_test, criterion, batch_size, evaluator
                )
            else:
                _, out, loss_train, score_train = evaluate(
                    model, data_eval, feats, vectors, labels, criterion, evaluator, idx_train
                )
                # Use criterion & evaluator instead of evaluate to avoid redundant forward pass
                loss_val = criterion(out[idx_val], labels[idx_val]).item()
                score_val = evaluator(out[idx_val], labels[idx_val])
                loss_test = criterion(out[idx_test], labels[idx_test]).item()
                score_test = evaluator(out[idx_test], labels[idx_test])

            logger.debug(
                f"Ep {epoch:3d} | loss: {loss:.4f} | s_train: {score_train:.4f} | s_val: {score_val:.4f} | s_test: {score_test:.4f}"
            )
            loss_and_score += [
                [
                    epoch,
                    loss_train,
                    loss_val,
                    loss_test,
                    score_train,
                    score_val,
                    score_test,
                ]
            ]

            if score_val >= best_score_val:
                best_epoch = epoch
                best_score_val = score_val
                state = copy.deepcopy(model.state_dict())
                count = 0
            else:
                count += 1

        if count == conf["patience"] or epoch == conf["max_epoch"]:
            break

    model.load_state_dict(state)
    if "MLP" in model.model_name:
        start = time.time()
        h_list, out, _, score_val = evaluate_mini_batch(
            model, feats, vectors, labels, criterion, batch_size, evaluator, idx_val
        )
        time_used = time.time()-start
        print("time_used: ", time_used)
    else:
        start = time.time()
        h_list, out, _, score_val = evaluate(
            model, data_eval, feats, vectors, labels, criterion, evaluator,
        )
        time_used = time.time()-start
        print("time_used: ", time_used)

    score_test = evaluator(out[idx_test], labels[idx_test])
    logger.info(
        f"Best valid model at epoch: {best_epoch: 3d}, score_val: {score_val :.4f}, score_test: {score_test :.4f}"
    )
    return h_list[0], out, score_val, score_test


def run_inductive(
    conf,
    model,
    g,
    feats,
    vectors,
    labels,
    indices,
    criterion,
    evaluator,
    optimizer,
    logger,
    loss_and_score,
):
    """
    Train and eval under the inductive setting.
    The train/valid/test split is specified by `indices`.
    idx starting with `obs_idx_` contains the node idx in the observed graph `obs_g`.
    idx starting with `idx_` contains the node idx in the original graph `g`.
    The model is trained on the observed graph `obs_g`, and evaluated on both the observed test nodes (`obs_idx_test`) and inductive test nodes (`idx_test_ind`).
    The input graph is assumed to be large. Thus, SAGE is used for GNNs, mini-batch is used for MLPs.

    idx_obs: Idx of nodes in the original graph `g`, which form the observed graph 'obs_g'.
    loss_and_score: Stores losses and scores.
    """

    set_seed(conf["seed"])
    device = conf["device"]
    batch_size = conf["batch_size"]
    obs_idx_train, obs_idx_val, obs_idx_test, idx_obs, idx_test_ind = indices

    feats = feats.to(device)
    vectors = vectors.to(device)
    labels = labels.to(device)
    obs_feats = feats[idx_obs]
    obs_vectors = vectors[idx_obs]
    obs_labels = labels[idx_obs]
    obs_g = g.subgraph(idx_obs)


    if "SAGE" in model.model_name:
        # Create dataloader for SAGE

        # Create csr/coo/csc formats before launching sampling processes
        # This avoids creating certain formats in each data loader process, which saves momory and CPU.
        obs_g.create_formats_()
        g.create_formats_()
        sampler = dgl.dataloading.MultiLayerNeighborSampler(
            [eval(fanout) for fanout in conf["fan_out"].split(",")]
        )
        obs_dataloader = dgl.dataloading.DataLoader(
            obs_g,
            obs_idx_train,
            sampler,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=conf["num_workers"],
        )

        sampler_eval = dgl.dataloading.MultiLayerFullNeighborSampler(1)
        obs_dataloader_eval = dgl.dataloading.DataLoader(
            obs_g,
            torch.arange(obs_g.num_nodes()),
            sampler_eval,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=conf["num_workers"],
        )
        dataloader_eval = dgl.dataloading.DataLoader(
            g,
            torch.arange(g.num_nodes()),
            sampler_eval,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=conf["num_workers"],
        )

        obs_data = obs_dataloader
        obs_data_eval = obs_dataloader_eval
        data_eval = dataloader_eval
    elif "MLP" in model.model_name:
        feats_train, vectors_train, labels_train = obs_feats[obs_idx_train], obs_vectors[obs_idx_train], obs_labels[obs_idx_train]
        feats_val, vectors_val, labels_val = obs_feats[obs_idx_val], obs_vectors[obs_idx_val], obs_labels[obs_idx_val]
        feats_test_tran, vectors_test_tran, labels_test_tran = (
            obs_feats[obs_idx_test],
            obs_vectors[obs_idx_test],
            obs_labels[obs_idx_test],
        )
        feats_test_ind, vectors_test_ind, labels_test_ind = feats[idx_test_ind], vectors[idx_test_ind], labels[idx_test_ind]

    else:
        obs_g = obs_g.to(device)
        g = g.to(device)

        obs_data = obs_g
        obs_data_eval = obs_g
        data_eval = g

    best_epoch, best_score_val, count = 0, 0, 0
    for epoch in range(1, conf["max_epoch"] + 1):
        if "SAGE" in model.model_name:
            _, loss = train_sage(
                model, obs_data, obs_feats, obs_vectors, obs_labels, criterion, optimizer
            )
        elif "MLP" in model.model_name:
            _, loss = train_mini_batch(
                model, feats_train, vectors_train, labels_train, batch_size, criterion, optimizer
            )
        else:
            _, loss = train(
                model,
                obs_data,
                obs_feats,
                obs_vectors,
                obs_labels,
                criterion,
                optimizer,
                obs_idx_train,
            )

        if epoch % conf["eval_interval"] == 0:
            if "MLP" in model.model_name:
                h_list_train, _, loss_train, score_train = evaluate_mini_batch(
                    model, feats_train, vectors_train, labels_train, criterion, batch_size, evaluator
                )
                h_list_val, _, loss_val, score_val = evaluate_mini_batch(
                    model, feats_val, vectors_val, labels_val, criterion, batch_size, evaluator
                )
                h_list_test_tran, _, loss_test_tran, score_test_tran = evaluate_mini_batch(
                    model,
                    feats_test_tran,
                    vectors_test_tran,
                    labels_test_tran,
                    criterion,
                    batch_size,
                    evaluator,
                )
                _, _, loss_test_ind, score_test_ind = evaluate_mini_batch(
                    model,
                    feats_test_ind,
                    vectors_test_ind,
                    labels_test_ind,
                    criterion,
                    batch_size,
                    evaluator,
                )
            else:
                _, obs_out, loss_train, score_train = evaluate(
                    model,
                    obs_data_eval,
                    obs_feats,
                    obs_vectors,
                    obs_labels,
                    criterion,
                    evaluator,
                    obs_idx_train,
                )
                # Use criterion & evaluator instead of evaluate to avoid redundant forward pass
                loss_val = criterion(
                    obs_out[obs_idx_val], obs_labels[obs_idx_val]
                ).item()
                score_val = evaluator(obs_out[obs_idx_val], obs_labels[obs_idx_val])
                loss_test_tran = criterion(
                    obs_out[obs_idx_test], obs_labels[obs_idx_test]
                ).item()
                score_test_tran = evaluator(
                    obs_out[obs_idx_test], obs_labels[obs_idx_test]
                )

                # Evaluate the inductive part with the full graph
                _, out, loss_test_ind, score_test_ind = evaluate(
                    model, data_eval, feats, vectors, labels, criterion, evaluator, idx_test_ind
                )
            logger.debug(
                f"Ep {epoch:3d} | loss: {loss:.4f} | s_train: {score_train:.4f} | s_val: {score_val:.4f} | s_tt: {score_test_tran:.4f} | s_ti: {score_test_ind:.4f}"
            )
            loss_and_score += [
                [
                    epoch,
                    loss_train,
                    loss_val,
                    loss_test_tran,
                    loss_test_ind,
                    score_train,
                    score_val,
                    score_test_tran,
                    score_test_ind,
                ]
            ]
            if score_val >= best_score_val:
                best_epoch = epoch
                best_score_val = score_val
                state = copy.deepcopy(model.state_dict())
                count = 0
            else:
                count += 1

        if count == conf["patience"] or epoch == conf["max_epoch"]:
            break

    model.load_state_dict(state)
    if "MLP" in model.model_name:
        h_list_val, obs_out, _, score_val = evaluate_mini_batch(
            model, obs_feats, obs_vectors, obs_labels, criterion, batch_size, evaluator, obs_idx_val
        )
        start = time.time()
        h_list_test_ind, out, _, score_test_ind = evaluate_mini_batch(
            model, feats, vectors, labels, criterion, batch_size, evaluator, idx_test_ind
        )
        time_used = time.time()-start
        print("time_used: ", time_used)

    else:
        h_list_val, obs_out, _, score_val = evaluate(
            model,
            obs_data_eval,
            obs_feats,
            obs_vectors,
            obs_labels,
            criterion,
            evaluator,
            obs_idx_val,
        )
        start = time.time()
        h_list, out, _, score_test_ind = evaluate(
            model, data_eval, feats, vectors, labels, criterion, evaluator, idx_test_ind
        )
        time_used = time.time()-start
        print("time_used: ", time_used)

    score_test_tran = evaluator(obs_out[obs_idx_test], obs_labels[obs_idx_test])

    out[idx_obs] = obs_out
    h_list[0][idx_obs] = h_list_val[0]

    logger.info(
        f"Best valid model at epoch: {best_epoch :3d}, score_val: {score_val :.4f}, score_test_tran: {score_test_tran :.4f}, score_test_ind: {score_test_ind :.4f}"
    )
    return h_list[0], out, score_val, score_test_tran, score_test_ind



"""
3. Distill
"""
  

def distill_run_transductive(
    conf,
    model,
    feats,
    vectors,
    labels,
    out_t_all,
    distill_indices,
    criterion_l,
    criterion_t,
    evaluator,
    optimizer,
    logger,
    loss_and_score,
    distill = None, 
    hidden_t_all = None,  
    Idx = None,
):
    """
    Distill training and eval under the transductive setting.
    The hard_label_train/soft_label_train/valid/test split is specified by `distill_indices`.
    The input graph is assumed to be large, and MLP is assumed to be the student model. Thus, node feature only and mini-batch is used.

    out_t: Soft labels produced by the teacher model.
    criterion_l & criterion_t: Loss used for hard labels (`labels`) and soft labels (`out_t`) respectively
    loss_and_score: Stores losses and scores.
    """

    set_seed(conf["seed"])
    device = conf["device"]
    batch_size = conf["batch_size"]
    lamb = conf["lamb"]
    idx_l, idx_t, idx_val, idx_test = distill_indices
    alpha = conf["alpha"]
    feats = feats.to(device)
    vectors = vectors.to(device)
    if hidden_t_all != None:
        hidden_t_all = hidden_t_all.to(device)
    labels = labels.to(device)
    out_t_all = out_t_all.to(device)

    feats_l, vectors_l, labels_l = feats[idx_l], vectors[idx_l], labels[idx_l]
    feats_t, vectors_t, out_t = feats[idx_t], vectors[idx_t], out_t_all[idx_t]
    if hidden_t_all != None:
        hidden_t = hidden_t_all[idx_t]
    feats_val, vectors_val, labels_val = feats[idx_val], vectors[idx_val], labels[idx_val]
    feats_test, vectors_test, labels_test = feats[idx_test], vectors[idx_test], labels[idx_test]

    best_epoch, best_score_val, count = 0, 0, 0
    for epoch in range(1, conf["max_epoch"] + 1):
        _, loss_l = train_mini_batch(
            model, feats_l, vectors_l, labels_l, batch_size, criterion_l, optimizer, 1
        )

        if distill != "hidden":
            _, loss_t = train_mini_batch(
                model, feats_t, vectors_t, out_t, batch_size, criterion_t, optimizer, lamb
            )
            loss = loss_l + loss_t
        else:
            Idx_t = ([val for val in Idx if val in idx_t])
            _, loss_t = train_mini_batch(
                model, feats_t, vectors_t, out_t, batch_size, criterion_t, optimizer, lamb
            )
            _, loss_h = train_mini_batch(model, feats_t, vectors_t, out_t, batch_size, criterion_t, optimizer, alpha ,distill, hidden_t, Idx_t
            )
            loss = loss_l + loss_t + loss_h

        if epoch % conf["eval_interval"] == 0:
            _, _, loss_l, score_l = evaluate_mini_batch(
                model, feats_l, vectors_l, labels_l, criterion_l, batch_size, evaluator
            )
            _, _, loss_val, score_val = evaluate_mini_batch(
                model, feats_val, vectors_val,  labels_val, criterion_l, batch_size, evaluator
            )
            _, _, loss_test, score_test = evaluate_mini_batch(
                model, feats_test, vectors_test, labels_test, criterion_l, batch_size, evaluator
            )

            logger.debug(
                f"Ep {epoch:3d} | loss: {loss:.4f} | s_l: {score_l:.4f} | s_val: {score_val:.4f} | s_test: {score_test:.4f}"
            )
            loss_and_score += [
                [epoch, loss_l, loss_val, loss_test, score_l, score_val, score_test]
            ]

            if score_val >= best_score_val:
                best_epoch = epoch
                best_score_val = score_val
                state = copy.deepcopy(model.state_dict())
                count = 0
            else:
                count += 1

        if count == conf["patience"] or epoch == conf["max_epoch"]:
            break

    model.load_state_dict(state)
    start = time.time()
    _, out, _, score_val = evaluate_mini_batch(
        model, feats, vectors, labels, criterion_l, batch_size, evaluator, idx_val
    )
    time_used = time.time()-start
    print("time_used: ", time_used)
    # Use evaluator instead of evaluate to avoid redundant forward pass
    score_test = evaluator(out[idx_test], labels_test)

    logger.info(
        f"Best valid model at epoch: {best_epoch: 3d}, score_val: {score_val :.4f}, score_test: {score_test :.4f}"
    )
    return out, score_val, score_test


def distill_run_inductive(
    conf,
    model,
    feats,
    vectors,
    labels,
    out_t_all,
    distill_indices,
    criterion_l,
    criterion_t,
    evaluator,
    optimizer,
    logger,
    loss_and_score,
    distill = None,
    hidden_t_all = None,
    Idx = None,
):
    """
    Distill training and eval under the inductive setting.
    The hard_label_train/soft_label_train/valid/test split is specified by `distill_indices`.
    idx starting with `obs_idx_` contains the node idx in the observed graph `obs_g`.
    idx starting with `idx_` contains the node idx in the original graph `g`.
    The model is trained on the observed graph `obs_g`, and evaluated on both the observed test nodes (`obs_idx_test`) and inductive test nodes (`idx_test_ind`).
    The input graph is assumed to be large, and MLP is assumed to be the student model. Thus, node feature only and mini-batch is used.

    idx_obs: Idx of nodes in the original graph `g`, which form the observed graph 'obs_g'.
    out_t: Soft labels produced by the teacher model.
    criterion_l & criterion_t: Loss used for hard labels (`labels`) and soft labels (`out_t`) respectively.
    loss_and_score: Stores losses and scores.
    """

    
        
    set_seed(conf["seed"])
    device = conf["device"]
    batch_size = conf["batch_size"]
    lamb = conf["lamb"]
    (
        obs_idx_l,
        obs_idx_t,
        obs_idx_val,
        obs_idx_test,
        idx_obs,
        idx_test_ind,
    ) = distill_indices
    alpha = conf["alpha"]
    feats = feats.to(device)
    vectors = vectors.to(device)
    labels = labels.to(device)
    out_t_all = out_t_all.to(device)

    if hidden_t_all != None:
        hidden_t_all = hidden_t_all.to(device)

    obs_feats = feats[idx_obs]
    obs_vectors = vectors[idx_obs]
    obs_labels = labels[idx_obs]
    obs_out_t = out_t_all[idx_obs]

    feats_l, vectors_l, labels_l = obs_feats[obs_idx_l], obs_vectors[obs_idx_l], obs_labels[obs_idx_l]
    feats_t, vectors_t, out_t = obs_feats[obs_idx_t], obs_vectors[obs_idx_t], obs_out_t[obs_idx_t] 

    
    
    if hidden_t_all != None:
        hidden_t = hidden_t_all[obs_idx_t]
    feats_val, vectors_val, labels_val = obs_feats[obs_idx_val], obs_vectors[obs_idx_val], obs_labels[obs_idx_val]
    feats_test_tran, vectors_test_tran, labels_test_tran = (
        obs_feats[obs_idx_test],
        obs_vectors[obs_idx_test], 
        obs_labels[obs_idx_test],
    )
    feats_test_ind, vectors_test_ind, labels_test_ind = feats[idx_test_ind], vectors[idx_test_ind], labels[idx_test_ind]

    best_epoch, best_score_val, count = 0, 0, 0
    for epoch in range(1, conf["max_epoch"] + 1):
        
        if distill != "hidden":
            _, loss_l = train_mini_batch(
                model, feats_l, vectors_l, labels_l, batch_size, criterion_l, optimizer, 1
            )
            _, loss_t = train_mini_batch(
                model, feats_t, vectors_t, out_t, batch_size, criterion_t, optimizer, lamb
            )
            #
            # print("lamb: ",lamb)
            loss = loss_l + loss_t 
        else:
            Idx_t = ([val for val in Idx if val in obs_idx_t])
            _, loss_l = train_mini_batch(
            model, feats_l, vectors_l, labels_l, batch_size, criterion_l, optimizer, 1
            )
            _, loss_t = train_mini_batch(
                model, feats_t, vectors_t, out_t, batch_size, criterion_t, optimizer, lamb
            )
            #print("lamb: ",lamb)
            _, loss_h = train_mini_batch(
                model, feats_t, vectors_t, out_t, batch_size, criterion_t, optimizer, alpha , distill, hidden_t, Idx_t
            )
            
            loss = loss_l + loss_t + loss_h

        if epoch % conf["eval_interval"] == 0:
            _, _, loss_l, score_l = evaluate_mini_batch(
                model, feats_l, vectors_l, labels_l, criterion_l, batch_size, evaluator
            )
            _, _, loss_val, score_val = evaluate_mini_batch(
                model, feats_val, vectors_val, labels_val, criterion_l, batch_size, evaluator
            )
            _, _, loss_test_tran, score_test_tran = evaluate_mini_batch(
                model,
                feats_test_tran,
                vectors_test_tran,
                labels_test_tran,
                criterion_l,
                batch_size,
                evaluator,
            )
            _, _, loss_test_ind, score_test_ind = evaluate_mini_batch(
                model,
                feats_test_ind,
                vectors_test_ind,
                labels_test_ind,
                criterion_l,
                batch_size,
                evaluator,
            )

            logger.debug(
                f"Ep {epoch:3d} | l: {loss:.4f} | s_l: {score_l:.4f} | s_val: {score_val:.4f} | s_tt: {score_test_tran:.4f} | s_ti: {score_test_ind:.4f}"
            )
            loss_and_score += [
                [
                    epoch,
                    loss_l,
                    loss_val,
                    loss_test_tran,
                    loss_test_ind,
                    score_l,
                    score_val,
                    score_test_tran,
                    score_test_ind,
                ]
            ]

            if score_val >= best_score_val:
                best_epoch = epoch
                best_score_val = score_val
                state = copy.deepcopy(model.state_dict())
                count = 0
            else:
                count += 1

        if count == conf["patience"] or epoch == conf["max_epoch"]:
            break

    model.load_state_dict(state)
    _, obs_out, _, score_val = evaluate_mini_batch(
        model, obs_feats, obs_vectors, obs_labels, criterion_l, batch_size, evaluator, obs_idx_val
    )
    start = time.time()
    _, out, _, score_test_ind = evaluate_mini_batch(
        model, feats, vectors, labels, criterion_l, batch_size, evaluator, idx_test_ind
    )
    time_used = time.time()-start
    print("time_used: ", time_used)
    # Use evaluator instead of evaluate to avoid redundant forward pass
    score_test_tran = evaluator(obs_out[obs_idx_test], labels_test_tran)
    out[idx_obs] = obs_out

    logger.info(
        f"Best valid model at epoch: {best_epoch: 3d} score_val: {score_val :.4f}, score_test_tran: {score_test_tran :.4f}, score_test_ind: {score_test_ind :.4f}"
    )
    return out, score_val, score_test_tran, score_test_ind

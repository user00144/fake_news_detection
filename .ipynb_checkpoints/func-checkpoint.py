fold = KFold(n_splits = 5, shuffle = True, random_state = seed)

def acc_fn(y_pred, y_true):
    accuracy = torch.eq(y_pred, y_true).sum().item()/len(y_pred)
    return accuracy

def train_func(model, dataloader, optim, loss_fn, acc_fn) :
    t_loss, t_acc = 0, 0
    model.train()
    for batch, (title_input_ids, context_input_ids, title_attention_mask, context_attention_mask, label) in enumerate(tqdm(dataloader)) :
        
        title_input_ids, context_input_ids, title_attention_mask, context_attention_mask, label = title_input_ids.to(device), context_input_ids.to(device), title_attention_mask.to(device), context_attention_mask.to(device), label.to(device) 
        
        outputs = model(title_input_ids = title_input_ids, title_attention_mask =title_attention_mask,
                        context_input_ids = context_input_ids, context_attention_mask = context_attention_mask).to(device)
        outputs = torch.softmax(outputs, dim=1).to(device)

        loss = loss_fn(outputs, label)
        acc = acc_fn(outputs.argmax(dim=1).to(device), label)

        optim.zero_grad()
        loss.backward()
        optim.step()

        t_loss += loss.item()
        t_acc += acc

    t_loss /= len(dataloader)
    t_acc /= len(dataloader)
    return t_loss, t_acc

def eval_func(model, dataloader, loss_fn, acc_fn) :
    e_loss, e_acc = 0, 0
    model.eval()
    with torch.no_grad() :
        for batch, (title_input_ids, context_input_ids, title_attention_mask, context_attention_mask, label) in enumerate(tqdm(dataloader)) :
            title_input_ids, context_input_ids, title_attention_mask, context_attention_mask, label = title_input_ids.to(device), context_input_ids.to(device), title_attention_mask.to(device), context_attention_mask.to(device), label.to(device) 
            outputs = model(title_input_ids = title_input_ids, title_attention_mask =title_attention_mask,
                        context_input_ids = context_input_ids, context_attention_mask = context_attention_mask).to(device)
            outputs = torch.softmax(outputs, dim=1).to(device)
            
            loss = loss_fn(outputs, label)
            acc = acc_fn(outputs.argmax(dim=1).to(device), label)

            e_loss += loss.item()
            e_acc += acc

        e_loss /= len(dataloader)
        e_acc /= len(dataloader)

    return e_loss, e_acc

def train(train_dataset , model, epochs, optim, loss_fn, acc_fn, patient, scheduler) :
    tot_tr_loss, tot_tr_acc = [], []
    tot_val_loss, tot_val_acc = [], []
    for i, (train_idx, val_idx) in enumerate(fold.split(train_dataset)) :
        min_val_loss = 2
        n_patience = 0
        train_ds = Subset(train_dataset, train_idx)
        val_ds = Subset(train_dataset, val_idx)

        train_dl = DataLoader(train_ds, batch_size = batch_size, shuffle = True, num_workers = 4)
        val_dl = DataLoader(val_ds, batch_size = batch_size, shuffle = False, num_workers = 4)

        for epoch in range(epochs) :
            print(f'==FOLD {i} / EPOCH {epoch}==')
            train_loss, train_acc = train_func(model, train_dl, optim, loss_fn, acc_fn)
            val_loss, val_acc = eval_func(model, val_dl, loss_fn, acc_fn)
            print(f'train_loss / acc : {train_loss} / {train_acc}%, val_loss / acc : {val_loss} / {val_acc}%')
            if np.round(min_val_loss, 5) > np.round(val_loss, 5) :
                min_val_loss = val_loss
                n_patience = 0
                print(f'Save the best params with val_loss:{val_loss:.4f}, val_acc:{val_acc:.4f}')
                torch.save(model.state_dict(), f'./best_w_{i}.pth')
            else :
                n_patience += 1

            scheduler.step(verbose = True)
            
            if n_patience >= patient :
                print('Early Stopping')
                break

            tot_tr_loss.append(train_loss)
            tot_tr_acc.append(train_acc)
            tot_val_loss.append(val_loss)
            tot_val_acc.append(val_acc)
            print(f'<<FOLD {i}>>')
            print(f"\t Train loss {np.mean(tot_tr_loss):.4f} | acc {np.mean(tot_tr_acc):.4f}")
            print(f"\t Valid loss {np.mean(tot_val_loss):.4f} | acc {np.mean(tot_val_acc):.4f}")


print("<<TEST>>")
test_loss, test_acc = [],[]
for i in range(5):
    clf.load_state_dict(torch.load(f'best_w_{i}.pth'))
    loss, acc = eval_fn(clf, test_dataloader, loss_fn, acc_fn)
    test_loss.append(loss)
    test_acc.append(acc)

print(f"Average  loss {np.mean(test_loss):.4f} | acc {np.mean(test_acc):.4f}")
print(f"Variance loss {np.var(test_loss):.4f} | acc {np.var(test_acc):.4f}")
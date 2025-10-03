#!/usr/bin/env python3
import os
import time
import math
import torch
from torch.nn.functional import softplus
from datetime import datetime
from tensorboardX import SummaryWriter
from nn import *
from training import *
from pyscf import gto, scf, dft
from pyscf.grad import rhf
import numpy as np



###################ZURIEL´s MODIFICATIONS#########################
from beta_scheduler import BetaScheduler
from smooth_transition_schedule import smooth_transition_schedule

###################ZURIEL´s MODIFICATIONS#########################
#without this, some things from torch don't
#work correctly in newer versions of python
import multiprocessing
multiprocessing.set_start_method('fork')

"""
################################################
################ INITIALIZATION ################
################################################
"""
# read arguments
args = parse_command_line_arguments()

#no restart directory specified
if args.restart is None:
    ID = generate_id() #generate "unique" id for the run (very unlikely that two runs will have the same ID)
    directory = datetime.utcnow().strftime("%Y-%m-%d_")+ID #generate directory name
    checkpoint_dir = os.path.join(directory, 'checkpoints') #checkpoint directory
    # create directories
    if not os.path.exists(directory):
        os.makedirs(directory)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    # write command line arguments to file (useful for reproducibility)
    with open(os.path.join(directory, 'args.txt'), 'w') as f:
        for key in args.__dict__.keys():
            if isinstance(args.__dict__[key], list): #special case for list input
                for entry in args.__dict__[key]:
                        f.write('--'+key+'='+str(entry)+"\n")
            else:
                f.write('--'+key+'='+str(args.__dict__[key])+"\n")
    checkpoint = None
    latest_checkpoint = 0
#restarts run from latest checkpoint
else:
    directory = args.restart #load directory name
    checkpoint_dir = os.path.join(directory, 'checkpoints') #checkpoint directory
    #load latest checkpoint
    checkpoint = torch.load(os.path.join(checkpoint_dir, 'latest_checkpoint.pth'), map_location='cpu')
    latest_checkpoint = checkpoint['epoch']
    ID   = checkpoint['ID'] #load ID
    args = checkpoint['args'] #overwrite args

#determine whether GPU is used for training
use_gpu = args.use_gpu and torch.cuda.is_available()

#load dataset(s)
print("loading " + args.dataset + "...")
dataset = HamiltonianDataset(args.dataset, dtype=args.dtype)

#split into train/valid/test
train_dataset, valid_dataset, test_dataset = seeded_random_split(dataset, [args.num_train, args.num_valid, len(dataset)-(args.num_train+args.num_valid)], seed=args.split_seed)

#save indices for splits
np.savez(os.path.join(directory, 'datasplits.npz'), train=train_dataset.indices, valid=valid_dataset.indices, test=test_dataset.indices)

#determine weights of different quantities for scaling loss
loss_weights = {}
loss_weights['density_matrix'] = args.density_matrix_weight
loss_weights['core_hamiltonian'] = args.core_hamiltonian_weight
loss_weights['overlap_matrix']   = args.overlap_matrix_weight
loss_weights['energy'] = args.energy_weight
loss_weights['forces'] = args.forces_weight

#if energies/forces are used for training, the extreme errors
#at the beginning of training usually lead to NaNs. For this
#reason gradients are only allowed to flow through loss terms
#if the MAE is smaller than a certain threshold.
max_errors = {}
max_errors['density_matrix'] = np.inf
max_errors['core_hamiltonian'] = np.inf
max_errors['overlap_matrix']   = np.inf
max_errors['energy'] = args.max_energy_error
max_errors['forces'] = args.max_forces_error

#prepare data loaders
train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True,  num_workers=args.num_workers, pin_memory=use_gpu, collate_fn=lambda batch: dataset.collate_fn(batch))
valid_data_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.valid_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=use_gpu, collate_fn=lambda batch: dataset.collate_fn(batch))
pred_chunk_size = np.max([args.valid_batch_size, args.train_batch_size])
#define model
if args.load_from is None:
    model = NeuralNetwork(
        orbitals             = dataset.orbitals,
        order                = args.order,
        num_features         = args.num_features,
        num_basis_functions  = args.num_basis_functions,
        num_modules          = args.num_modules,
        num_residual_pre_x   = args.num_residual_pre_x,
        num_residual_post_x  = args.num_residual_post_x,
        num_residual_pre_vi  = args.num_residual_pre_vi,
        num_residual_pre_vj  = args.num_residual_pre_vj,
        num_residual_post_v  = args.num_residual_post_v,
        num_residual_output  = args.num_residual_output,
        num_residual_pc      = args.num_residual_pc,
        num_residual_pn      = args.num_residual_pn,
        num_residual_ii      = args.num_residual_ii,
        num_residual_ij      = args.num_residual_ij,
        num_residual_full_ii = args.num_residual_full_ii,
        num_residual_full_ij = args.num_residual_full_ij,
        num_residual_core_ii = args.num_residual_core_ii,
        num_residual_core_ij = args.num_residual_core_ij,
        num_residual_over_ij = args.num_residual_over_ij,
        basis_functions      = args.basis_functions,
        cutoff               = args.cutoff,
        activation           = args.activation,
        restricted_scheme    = args.restricted_scheme,
        system_net_charge    = args.system_net_charge
    )
else:
    model = NeuralNetwork(load_from=args.load_from)



#determine what should be calculated based on loss weights
tmp = (loss_weights['energy'] > 0) or (loss_weights['forces'] > 0)
model.calculate_density_matrix =  (loss_weights['density_matrix'] > 0) or tmp
model.calculate_core_hamiltonian =  (loss_weights['core_hamiltonian'] > 0) or tmp
model.calculate_overlap_matrix   = ((loss_weights['overlap_matrix']   > 0) or tmp) and not args.orthonormal_basis
model.calculate_energy  = loss_weights['energy'] > 0
model.calculate_forces  = loss_weights['forces'] > 0
Norb = model.Norb
num_electrons = torch.sum(model.Z)-model.system_net_charge
print("num_electrons= ", num_electrons)
restricted = model.restricted_scheme

#convert the model to the correct dtype
model.to(args.dtype)


#send model to GPU (if use_gpu is True)
if use_gpu:
    model.cuda()

#if there are multiple GPUs, wrap the model in DataParallel
#"module" is used whenever direct access is needed, e.g. for parameters,
#whereas "model" may be DataParallel and is used for inference only
if use_gpu and torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
    module = model.module
else:
    module = model

#for keeping an exponential moving average of the model parameters (usually leads to better models)
if args.use_parameter_averaging:
    exponential_moving_average = ExponentialMovingAverage(module, decay=args.ema_decay, start_epoch=args.ema_start_epoch)
else:
    exponential_moving_average = None

#build list of parameters to optimize (with or without weight decay)
parameters = []
weight_decay_parameters = []
for name, param in module.named_parameters():
    if 'weight' in name and not 'radial_fn' in name and not 'embedding' in name:
        weight_decay_parameters.append(param)
    else:
        parameters.append(param)

parameter_list = [
    {'params': parameters},
    {'params': weight_decay_parameters, 'weight_decay': float(args.weight_decay)}]

#choose optimizer
if  args.optimizer == 'adam':     #Adam
    print("using Adam optimizer")
    optimizer = torch.optim.AdamW(parameter_list,  lr=args.learning_rate, eps=args.epsilon, betas=(args.beta1, args.beta2), weight_decay=0.0)
elif args.optimizer == 'amsgrad':  #AMSGrad
    print("using AMSGrad optimizer")
    optimizer = torch.optim.AdamW(parameter_list,  lr=args.learning_rate, eps=args.epsilon, betas=(args.beta1, args.beta2), weight_decay=0.0, amsgrad=True)
elif args.optimizer == 'sgd': #Stochastic Gradient Descent
    print("using Stochastic Gradient Descent optimizer")
    optimizer = torch.optim.SGD(parameter_list,  lr=args.learning_rate, momentum=args.momentum, weight_decay=0.0)

#initialize Lookahead
if args.lookahead_k > 0:
    optimizer = Lookahead(optimizer, k=args.lookahead_k)

#learning rate scheduler (decays learning rate if validation loss plateaus)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=args.decay_factor, patience=args.decay_patience)

#restore state from checkpoint
if checkpoint is not None: #no checkpoint is specified
    step  = checkpoint['step']
    epoch = checkpoint['epoch']
    best_errors = checkpoint['best_errors']
    valid_errors = checkpoint['valid_errors']
    module.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    if exponential_moving_average is not None:
        checkpoint_ema = checkpoint['exponential_moving_average']
        for key in exponential_moving_average.ema.keys():
            with torch.no_grad():
                exponential_moving_average.ema[key].data.copy_(checkpoint_ema[key].data)
#or initialize step/epoch to 0 and errors to infinity
else:
    step = 0
    epoch = 0
    best_errors = empty_error_dict(loss_weights, fill_value=math.inf)
    valid_errors = empty_error_dict(loss_weights, fill_value=math.inf)

#create summary writer for tensorboard
summary = SummaryWriter(logdir=os.path.join(directory, 'logs'), purge_step=step)



###############ZURIEL´s MODIFICATIONS###################################


device=next(model.parameters()).device
dtype=next(model.parameters()).dtype
transition_epochs = 100

#(DFTDIFF, OSE, TRE, IDE, MOE)
initial_weights =  [1.00, 0.0, 0.0, 0.0, 0.0]
second_weights = [0.50, 0.0, 0.0, 0.0, 0.0]
third_weights = [0.5, 0.05, 0.05, 0.05, 0.05]


beta_scheduler = BetaScheduler(
                      beta_init=initial_weights,
                      beta_target=second_weights,
                      schedules=[
                                          smooth_transition_schedule(epoch, epoch+transition_epochs),  
                                          smooth_transition_schedule(epoch, epoch+transition_epochs),
                                          smooth_transition_schedule(epoch, epoch+transition_epochs),
                                          smooth_transition_schedule(epoch, epoch+transition_epochs),
                                          smooth_transition_schedule(epoch,epoch+ transition_epochs)
                          
                                          
                                                                                                                                                                ],
                      device=device,
                      dtype=args.dtype)

  

# Initialize mutable beta tensor
beta, progress = beta_scheduler(0)


second_stage_lr_offset = 1e-40
scf_iter_log_lr = 5e-50

model.downstream_mode= False
scf_evals_active = False
scf_evals_started = False
model.purification_iterations=3
#########################################################################


"""
###############################################
################ TRAINING LOOP ################
###############################################
"""
if use_gpu:
    print("Training on "+str(torch.cuda.device_count())+" GPUs:")
else:
    print("Training on the CPU:")

#initialize train metrics
if args.use_gradient_clipping:
    gradient_norm = 0
train_errors = empty_error_dict(loss_weights) #reset train error metrics
train_batch_num = -1
#initialize state
model.train()
train_iterator = iter(train_data_loader)
new_valid = False
new_best = False

start_time = time.time()
while step < args.max_steps+1:
    #get the next batch
    try:
        data = next(train_iterator)
    except StopIteration:
        epoch += 1
        
        beta, progress= beta_scheduler(epoch)
        if optimizer.param_groups[0]['lr'] <= second_stage_lr_offset:
            print("Now in Stage Two!!!")
            beta_scheduler = BetaScheduler(
                      beta_init=second_weights,
                      beta_target=third_weights,
                      schedules=[
                                          smooth_transition_schedule(epoch, epoch + transition_epochs),  
                                          smooth_transition_schedule(epoch, epoch + transition_epochs),
                                          smooth_transition_schedule(epoch, epoch +transition_epochs),
                                          smooth_transition_schedule(epoch, epoch + transition_epochs),
                                          smooth_transition_schedule(epoch, epoch + transition_epochs)
                                          
                                                                                                                                                                ],
                      device=device,
                      dtype=args.dtype)
        if not scf_evals_started and optimizer.param_groups[0]['lr'] <= scf_iter_log_lr:
            scf_evals_started = True
            print("Evaluating SCF iterations from now on!!!")
            

            
       
        train_iterator = iter(train_data_loader)
        continue
    train_batch_num += 1

    #send data to GPU
    if use_gpu:
        for key in data.keys():
            if isinstance(data[key], torch.Tensor):
                data[key] = data[key].cuda()

    #zero the parameter gradients
    optimizer.zero_grad()

    #with torch.autograd.set_detect_anomaly(True): #TODO!!! TURN THIS OFF AGAIN

    #forward step
    assert "overlap_matrix" in data, "FORGOT TO INCLUDE OVERLAP MATRIX IN TRAINING DATA!!"
    S = data["overlap_matrix"]
    predictions= model(R=data['positions'], S = S)
    if "density_matrix" in predictions:
        train_predicted_dms = torch.mean((1.0*predictions["density_matrix"]).detach(), axis = 0)
        train_predicted_dms_min, train_predicted_dms_max = torch.min(train_predicted_dms).item(),  torch.max(train_predicted_dms).item()
    #compute error metrics
    errors, train_eigvals, P_rep = compute_error_dict(predictions, data, loss_weights, max_errors, beta, num_electrons, chemical_dict, model.purification_iterations, restricted = restricted, main_split_Q_vs_AO = float(beta[0]))
    if isinstance(train_eigvals, torch.Tensor):
        train_eigvals = torch.mean(train_eigvals, axis = 0)
        min_train_eigvals, max_train_eigvals = torch.min(train_eigvals).item(), torch.max(train_eigvals).item()
    else:
        min_train_eigvals, max_train_eigvals = train_eigvals, train_eigvals
    
        

    #backward step
    errors['loss'].backward()

    #apply gradient clipping
    if args.use_gradient_clipping:
        norm = torch.nn.utils.clip_grad_norm_(module.parameters(), args.clip_norm)
        gradient_norm += (norm - gradient_norm)/(train_batch_num+1)

    #optimization step
    optimizer.step()

    #update parameter averages
    if args.use_parameter_averaging:
        exponential_moving_average(epoch)

    #update train_errors (running average)
    for key in train_errors.keys():
        train_errors[key] += (errors[key].item() - train_errors[key])/(train_batch_num+1)

    #run validation each validation_interval
    if step%args.validation_interval == 0:
        #this is a signal to the summary writer
        new_valid = True

        #swap to exponentially averaged parameters for validation
        if args.use_parameter_averaging:
            exponential_moving_average.swap()

        #run once over the validation set
        valid_errors = empty_error_dict(loss_weights) #reset valid error metrics
        model.eval() #sets model to evaluation mode
        for valid_batch_num, data in enumerate(valid_data_loader):
            #send data to GPU
            if use_gpu:
                for key in data.keys():
                    if isinstance(data[key], torch.Tensor):
                        data[key] = data[key].cuda()
                        
            assert "overlap_matrix" in data, "FORGOT TO INCLUDE OVERLAP MATRIX IN VALIDATION DATA!!"
            S = data["overlap_matrix"]
            predictions= model(R=data['positions'], S = S)

            
            if "density_matrix" in predictions:
                valid_predicted_dms = torch.mean((1.0*predictions["density_matrix"]).detach(), axis = 0)
                valid_predicted_dms_min, valid_predicted_dms_max = torch.min(valid_predicted_dms).item(),  torch.max(valid_predicted_dms).item()
                 #compute error metrics
            errors, valid_eigvals, P_rep = compute_error_dict(predictions, data, loss_weights, max_errors, beta, num_electrons, chemical_dict, model.purification_iterations, restricted = restricted, main_split_Q_vs_AO = float(beta[0]))
            if isinstance(valid_eigvals, torch.Tensor):
                valid_eigvals = torch.mean(valid_eigvals, axis = 0)
                min_valid_eigvals, max_valid_eigvals = torch.min(valid_eigvals).item(), torch.max(valid_eigvals).item()
            else:
                min_valid_eigvals, max_valid_eigvals = valid_eigvals, valid_eigvals

            #update valid_errors (running average)
            for key in valid_errors.keys():
                valid_errors[key] += (errors[key].item() - valid_errors[key])/(valid_batch_num+1)

        #pass validation loss to learning rate scheduler
        scheduler.step(metrics=valid_errors['loss'])

        #save if it outperforms previous best
        if valid_errors['loss'] < best_errors['loss']:
            new_best = True
            best_errors = valid_errors
            module.save(os.path.join(directory, 'best_'+str(ID)+'.pth'))
            #construct message for logging
            message = ''
            for key in best_errors.keys():
                message += key + ': %.6e' % best_errors[key] + '\n'
            summary.add_text('best models', message, step)

        #swap back to original parameters for training
        if args.use_parameter_averaging:
            exponential_moving_average.swap()

        #set model back to training mode
        model.train()

    #write summary to console
    if step%args.summary_interval == 0:
        #write error summaries
        for key in train_errors.keys():
            summary.add_scalar(key+'/train', train_errors[key], step)

        if new_valid:
            for key in valid_errors.keys():
                summary.add_scalar(key+'/valid', valid_errors[key], step)
            new_valid = False

        if new_best:
            for key in best_errors.keys():
                summary.add_scalar(key+'/best', best_errors[key], step)
            new_best = False

        if args.use_gradient_clipping:
            summary.add_scalar('gradient/norm', gradient_norm, step)
            gradnorm = (1.0*gradient_norm).detach().item()

        #write summaries for scalar model parameters (always)
        summary.add_scalar('rbf/alpha', softplus(module.radial_basis_functions._alpha), step)

        #write optional summaries for model parameters
        if args.write_parameter_summaries:
            for name, param in module.named_parameters():
                splitted_name = name.split('.', 1)
                if len(splitted_name) > 1:
                    first, last = splitted_name
                else:
                    first = 'nn'
                    last = splitted_name[0]
                if param.numel() > 1 and param.requires_grad: #only tensors get written as histogram
                    summary.add_histogram(first+'/'+last, param.clone().cpu().data.numpy(), step)

 ############################ZURIEL´s MODIFICATION###########################################################
  
 
 
 
# Print training progress
        
        scf_iter_string = "SCF iterations metrics (Valid): [Not Available Yet]"
        if (scf_evals_started and scf_evals_active) and P_rep is not None:
            P_dfts1 = (1.0*P_rep).detach()
            S_real = (1.0*data["overlap_matrix"]).detach()
            niters = scf_iterations(P_dfts1, data["positions"], Norb)
            iters_mean, iters_min, iters_max = np.mean(niters), np.min(niters), np.max(niters)
            scf_iter_string = f"SCF iterations metrics (Valid): min={iters_min:.2f}, max={iters_max:.2f}, mean={iters_mean:.2f}"
    
        scf_evals_active = not scf_evals_active
 

        
        
        progress_string = "\n"
        progress_string += f"Step {str(step).zfill(len(str(args.max_steps)))}/{args.max_steps} | Epoch: {epoch:<6d}\n"
        progress_string += "-" * 80 + "\n"
        progress_string += f"{'':<25}{'Train':>15}{'Valid':>15}{'Best':>15}\n"
        progress_string += "-" * 80 + "\n"
        
        # Total loss
        progress_string += f"{'Total Loss:':<25}{train_errors['loss']:>15.6e}{valid_errors['loss']:>15.6e}{best_errors['loss']:>15.6e}\n"
        
        # Detailed metrics
        for key in loss_weights:
           
          
            if loss_weights[key] > 0:
                progress_string += f"\n[{key.upper()}]\n"
                progress_string += f"{'MAE:':<25}{train_errors[key + '_mae']:>15.6e}{valid_errors[key + '_mae']:>15.6e}{best_errors[key + '_mae']:>15.6e}\n"
                progress_string += f"{'RMSE:':<25}{train_errors[key + '_rmse']:>15.6e}{valid_errors[key + '_rmse']:>15.6e}{best_errors[key + '_rmse']:>15.6e}\n"
               
                if key == "density_matrix":
                
                    if float(beta[1]) >0.0: 
                        progress_string += f"{'OCC. SPECTRUM ERROR(OSE):':<25}{train_errors[key + '_ose']:>15.6e}{valid_errors[key + '_ose']:>15.6e}{best_errors[key + '_ose']:>15.6e}\n"
                    if float(beta[2]) >0.0:    
                        progress_string += f"{'IDEMPOTENCY ERROR(IDE):':<25}{train_errors[key + '_ide']:>15.6e}{valid_errors[key + '_ide']:>15.6e}{best_errors[key + '_ide']:>15.6e}\n"

                    if float(beta[3]) >0.0:     
                        progress_string += f"{'TRACE ERROR (TRE):':<25}{train_errors[key + '_tre']:>15.6e}{valid_errors[key + '_tre']:>15.6e}{best_errors[key + '_tre']:>15.6e}\n"
                    if float(beta[4]) >0.0:     
                        progress_string += f"{'MOREAU ERROR (MOE):':<25}{train_errors[key + '_moe']:>15.6e}{valid_errors[key + '_moe']:>15.6e}{best_errors[key + '_moe']:>15.6e}\n"
                          
                    
                
        progress_string += "-" * 80 + "\n"
        print(progress_string)
        
     
        # Time elapsed
        end_time = time.time()
        print(f"Time elapsed: {end_time - start_time:.4f} seconds")
        # Current wall-clock time in YYYY-MM-DD HH:MM:SS format
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"Current Time: {now_str}")
        print(f"Current Learning Rate: {optimizer.param_groups[0]['lr']:.8f}")
        print(f"Current Gradient Norm {gradnorm:.8f}")
        start_time = end_time
        
        
        # Beta values
        beta_str = ", ".join([f"{b:.4f}" for b in beta.tolist()])
        progress_str = ", ".join([f"{p:.4f}" for p in progress.tolist()])
        print(f"Epoch {epoch} | Beta: [{beta_str}] | Progress: [{progress_str}]")
   
        
        # Further metrics (min/max)
        print("-" * 80)
        print("Further Training Metrics:")
        print(f"Density Matrix Range (Train): min={train_predicted_dms_min:.6e}, max={train_predicted_dms_max:.6e}")
        if isinstance(min_train_eigvals, float) and isinstance(max_train_eigvals, float):
            print(f"Eigenvalue Range (Train): min={min_train_eigvals:.6e}, max={max_train_eigvals:.6e}")
        else:
            print("Eigenvalue Range (Train): [Not Available]")
        
        print("-" * 80)
        print("Further Validation Metrics:")
        print(f"Density Matrix Range (Valid): min={valid_predicted_dms_min:.6e}, max={valid_predicted_dms_max:.6e}")
        if isinstance(min_valid_eigvals, float) and isinstance(max_valid_eigvals, float):
            print(f"Eigenvalue Range (Valid): min={min_valid_eigvals:.6e}, max={max_valid_eigvals:.6e}")
        else:
            print("Eigenvalue Range (Valid): [Not Available]")
        print(scf_iter_string )
        print("-" * 80)
         
 
  ############################ZURIEL´s MODIFICATION###########################################################

 
 

        #reset train metrics
        if args.use_gradient_clipping:
            gradient_norm = 0
        train_errors = empty_error_dict(loss_weights) #reset train error metrics
        train_batch_num = -1

    #increment step counter
    step += 1

    #save checkpoint (always the last step)
    if step%args.checkpoint_interval == 0:
        #move latest checkpoint (so it is not overwritten)
        if os.path.isfile(os.path.join(checkpoint_dir, 'latest_checkpoint.pth')):
            os.rename(os.path.join(checkpoint_dir, 'latest_checkpoint.pth'), os.path.join(checkpoint_dir, 'checkpoint_'+str(latest_checkpoint).zfill(10)+'.pth'))
        latest_checkpoint = step

        #overwrite latest checkpoint
        torch.save({
            'ID': ID,
            'args': args,
            'step': step,
            'epoch': epoch,
            'best_errors': best_errors,
            'valid_errors': valid_errors,
            'model_state_dict': module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'exponential_moving_average': (exponential_moving_average.ema if args.use_parameter_averaging else None)
            }, os.path.join(checkpoint_dir, 'latest_checkpoint.pth'))
        summary.add_text('checkpoints', 'saved checkpoint', step)

        #remove oldest checkpoints
        if args.keep_checkpoints >= 0: #for negative arguments, all checkpoints are kept
            for file in os.listdir(checkpoint_dir):
                if file.startswith("checkpoint") and file.endswith('.pth'):
                    checkpoint_step = int(file.split('.pth')[0].split('_')[-1])
                    if checkpoint_step < step - args.checkpoint_interval*args.keep_checkpoints:
                        filename = os.path.join(checkpoint_dir, file)
                        if os.path.isfile(filename):
                            os.remove(filename)

    #decide whether to stop the run based on learning rate
    stop_training = True
    for param_group in optimizer.param_groups:
        stop_training = stop_training and (param_group['lr'] < args.stop_at_learning_rate)
    if stop_training:
        print("Learning rate is smaller than "+str(args.stop_at_learning_rate)+"! Training stopped.")
        break

#close summary writer
summary.close()
"""
######################## POST TRAINING COMPUTATIONS ###########################################
"""

print("NOW COMPUTING AND SAVING PREDICTIONS!!!")
print(f"Calculating on {torch.cuda.device_count()} GPU(s)")

# --- Prepare ---
keys = [key for key in loss_weights if loss_weights[key] > 0.0]
dm_ind = keys.index("density_matrix")

bulk_data = torch.load("num_bulk_data.pth", map_location=device)
model.eval(); model.downstream_mode= True  # set model to eval mode

# Ensure double precision
for param in model.parameters():
    param.data = param.data.to(args)
model = model.to(torch.float64)

# --- Utility function for batch prediction ---
def predict_batch(position_list, overlap_matrix_list, keys = keys, model=model, pred_chunk_size = pred_chunk_size):
    """
    Accepts a list of position tensors [N_data x (N_atoms x 3)]
    Returns dict of predictions per key: {key: list of arrays}
    """
    with torch.no_grad():
        assert len(overlap_matrix_list) == len(position_list), "TO PREDICT YOU NEED ONE S MATRIX FOR EACH POSITION VECTOR"
        R = torch.stack(position_list).to(model.device).to(torch.float64)  # [B, N_atoms, 3]
        S = torch.stack(overlap_matrix_list).to(model.device).to(torch.float64) # [B, N_orbitals, N_orbitals]
        predictions = {k : [] for k in keys}
        offset = 0
        while offset < len(overlap_matrix_list):
            r_chunk, s_chunk = R[offset:offset+pred_chunk_size], S[offset:offset+pred_chunk_size]
            preds = model(r_chunk, s_chunk)
            for k in keys:
                predictions[k].extend(transform(preds[k].detach().cpu().numpy(),convention= "phipy"))
            offset += pred_chunk_size
        return [predictions[k] for k in keys]
                
           
                
        
       

# --- Collect indices ---
train_indices = train_dataset.indices
valid_indices = valid_dataset.indices
test_indices  = test_dataset.indices

# --- Helper to extract and predict ---
def extract_positions(indices):
    return [bulk_data[i]["positions"] for i in indices]
    
def extract_overlap_matrices(indices):
    return [bulk_data[i]["overlap_matrix"] for i in indices]


def extract_real_values(indices, keys):
    return [[transform(bulk_data[i][k].cpu().numpy(), convention = "phipy") for i in indices] for k in keys]

def predict_for_indices(indices):
    positions = extract_positions(indices)
    overlap_matrices =  extract_overlap_matrices(indices)
    return predict_batch(positions, overlap_matrices)

# --- Run predictions ---
useful_data = {
    "real": {
        "train": extract_real_values(train_indices, keys),
        "valid":  extract_real_values(valid_indices, keys),
        "test":  extract_real_values(test_indices,  keys)
       
    },
    "predicted": {
        "train": predict_for_indices(train_indices),
        "valid":  predict_for_indices(valid_indices),
        "test":  predict_for_indices(test_indices)
    },
    
     "positions": {
        "train": extract_positions(train_indices),
        "valid": extract_positions(valid_indices),
        "test":  extract_positions(test_indices)}}

# --- Save output ---
outputs_path = os.path.join(directory, f'useful_data{ID}.npy')

np.save(outputs_path, useful_data)
with open(os.path.join(directory, f'key_order_{ID}.txt'), "w") as f:
    f.write(f"key order: {keys}")


"""
######################## NOW GETTING EVALUACTION LOGS!! ###########################################
"""
print("NOW GETTING EVALUACTION LOGS!! ")



rs_valid = useful_data["positions"]["valid"]
dms_valid = useful_data["predicted"]["valid"][dm_ind]

if "dms_valid" not in globals() or "rs_valid" not in globals():
    raise RuntimeError("Please define dms_valid (N,nao,nao) and rs_valid (N,nat,3 in Bohr) in this session.")

# Basic shape checks vs the selected composition
if rs_valid.shape[1] != len(chemical_symbols) or rs_valid.shape[2] != 3:
    raise ValueError(f"rs_valid shape {rs_valid.shape} incompatible with chemical_symbols {chemical_symbols}")

# Run the dataset evaluation and write a single .txt file
reports, summary = evaluate_dataset_to_txt(
    dms_valid, rs_valid, chemical_symbols,
    outfile=f"benchmarks_summary_{mol_sym}.txt",
    enforce_trace=False,          # raw P used for direct; trace-correct only for SCF seed if True
    basis= basis,
    newton_rescue=False,
    verbose=0,
    print_each=True,
    unit=unit,
    charge=charge,
    spin=spin
)
print(f"Wrote: benchmarks_summary_{mol_sym}.txt")













print(f"ALL COMPLETED! Predictions saved to: {outputs_path}")

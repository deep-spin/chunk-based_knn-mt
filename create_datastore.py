import logging
import os
import sys
from itertools import chain
import argparse
import numpy as np
import faiss
import time
import torch
from tqdm import tqdm

from fairseq import checkpoint_utils, distributed_utils, options, utils
from fairseq.logging import metrics, progress_bar

from sklearn.decomposition import PCA
import pickle


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("fairseq_cli.validate")


def main(args, override_args=None):
    start = time.time()
    utils.import_user_module(args)

    assert(args.max_tokens is not None or args.batch_size is not None
    ), "Must specify batch size either with --max-tokens or --batch-size"

    use_fp16 = args.fp16
    use_cuda = torch.cuda.is_available() and not args.cpu

    if use_cuda:
        torch.cuda.set_device(args.device_id)

    if override_args is not None:
        overrides = vars(override_args)
        overrides.update(eval(getattr(override_args, "model_overrides", "{}")))
    else:
        overrides = None

    # Load ensemble
    # the task is build based on the checkpoint
    logger.info("loading model(s) from {}".format(args.path))
    models, model_args, task = checkpoint_utils.load_model_ensemble_and_task([args.path],arg_overrides=overrides,
        suffix=getattr(args, "checkpoint_suffix", ""),)
    model = models[0]

    # Move models to GPU
    for model in models:
        if use_fp16:
            model.half()
        if use_cuda:
            model.cuda()

    # Print args
    logger.info(model_args)

    # --- check save data store , add by
    import numpy as np
    
    if args.dstore_fp16:
        print('Saving fp16')
        dstore_keys = np.memmap(args.dstore_mmap + '/keys.npy', dtype=np.float16, mode='w+',shape=(args.dstore_size, args.decoder_embed_dim))
        dstore_vals = np.memmap(args.dstore_mmap + '/vals.npy', dtype=np.int, mode='w+',shape=(args.dstore_size, args.chunk_size))
    else:
        print('Saving fp32')
        dstore_keys = np.memmap(args.dstore_mmap + '/keys.npy', dtype=np.float32, mode='w+',shape=(args.dstore_size, args.decoder_embed_dim))
        dstore_vals = np.memmap(args.dstore_mmap + '/vals.npy', dtype=np.int, mode='w+',shape=(args.dstore_size, args.chunk_size))
    
    
    if args.save_decoder_states:
        if args.pca_decoder_states>0:
            dstore_decoder_states = np.memmap(args.dstore_mmap + '/decoder_states_pca_'+str(args.pca_decoder_states), dtype=np.float16, mode='w+',shape=(int(args.dstore_size*3), args.decoder_embed_dim))
        else:
            dstore_decoder_states = np.memmap(args.dstore_mmap + '/decoder_states', dtype=np.float16, mode='w+',shape=(int(args.dstore_size*3), args.decoder_embed_dim))
    

    dstore_idx = 0
    data_idx = 1
    decoder_states_size = 0
    for subset in args.valid_subset.split(","):
        try:
            model_args.dataset.required_seq_len_multiple = 1
            model_args.task.load_alignments = False
            task.load_dataset(subset, combine=False, epoch=data_idx)
            data_idx = data_idx + 1
            dataset = task.dataset(subset)
        except KeyError:
            raise Exception("Cannot find dataset: " + subset)

        args.batch_size=1

        # Initialize data iterator
        itr = task.get_batch_iterator(dataset=dataset, max_tokens=args.max_tokens, max_sentences=args.batch_size,
            max_positions=utils.resolve_max_positions(task.max_positions(), *[m.max_positions() for m in models],),
            ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
            required_batch_size_multiple=args.required_batch_size_multiple, seed=args.seed,
            num_shards=args.distributed_world_size, shard_id=args.distributed_rank, num_workers=args.num_workers,
            data_buffer_size=args.data_buffer_size,).next_epoch_itr(False)
        
        progress = progress_bar.progress_bar(itr, log_format=args.log_format, log_interval=args.log_interval,
           prefix=f"valid on '{subset}' subset", default_log_format=("tqdm" if not args.no_progress_bar else "simple"),)

        log_outputs = []
        with torch.no_grad():
            model.eval()
            
            for i, sample in enumerate(progress):
                sample = utils.move_to_cuda(sample) if use_cuda else sample
                features = task.forward_and_get_hidden_state_step(sample, model) # [B, T, H]
                target = sample['target']  # [B, T]

                #print(sample)

                # get useful parameters
                batch_size = target.size(0)
                seq_len = target.size(1)
                pad_idx = task.target_dictionary.pad()
                target_mask = target.ne(pad_idx)  # [B, T]

                if args.chunk_size>1:
                    if target.size(1)>=args.chunk_size:
                        for t in range(target.size(1)-(args.chunk_size-1)):
                            if t==0:
                                targets = target[:,t:t+args.chunk_size]
                            else:
                                targets = torch.cat([targets, target[:,t:t+args.chunk_size]],1)

                        for v in range(args.chunk_size-1):
                            targets = torch.cat([targets, torch.cat([target[:,t+1+v:t+args.chunk_size], torch.ones(target.size(0),1+v).cuda()],-1)],1)
                            
                    else:
                        for t in range(target.size(1)):
                            if t==0:
                                targets = torch.cat([target[:,t:t+args.chunk_size], torch.ones(target.size(0), args.chunk_size-target.size(1)+t).cuda()],-1)
                            else:
                                targets = torch.cat([targets, torch.cat([target[:,t:t+args.chunk_size], torch.ones(target.size(0), args.chunk_size-target.size(1)+t).cuda()],-1)],1)

                    target = targets.view(batch_size * seq_len, args.chunk_size)
                    target_mask = target_mask.contiguous().view(batch_size * seq_len)
                    non_pad_index = target_mask.nonzero().squeeze(-1)  # [n_count]
                    target = target.index_select(dim=0, index=non_pad_index)  # [n_count]
                    features = features.contiguous().view(batch_size * seq_len, -1)
                    features = features.index_select(dim=0, index=non_pad_index)  # [n_count, feature size]


                else:
                    # remove the pad tokens and related hidden states
                    target = target.view(batch_size * seq_len)
                    target_mask = target_mask.view(batch_size * seq_len)

                    non_pad_index = target_mask.nonzero().squeeze(-1)  # [n_count]
                    target = target.index_select(dim=0, index=non_pad_index).unsqueeze(-1)  # [n_count]

                    features = features.contiguous().view(batch_size * seq_len, -1)
                    features = features.index_select(dim=0, index=non_pad_index)  # [n_count, feature size]

                if args.chunk_size>1 and args.save_decoder_states:
                    pad_feature = torch.ones(1,features.size(1)).cuda()
                    pad_feature[:] = 100

                    decoder_states = torch.cat([features, pad_feature.repeat(args.chunk_size-1,1)])

                # save to the dstore
                current_batch_count = target.size(0)
                if dstore_idx + current_batch_count > args.dstore_size:
                    reduce_size = args.dstore_size - dstore_idx
                    features = features[:reduce_size]
                    target = target[:reduce_size]
                else:
                    reduce_size = current_batch_count

                
                if args.dstore_fp16:
                    dstore_keys[dstore_idx:reduce_size + dstore_idx] = features.detach().cpu().numpy().astype(np.float16)
                    dstore_vals[dstore_idx:reduce_size + dstore_idx] = target.cpu().numpy().astype(np.int)
                else:
                    dstore_keys[dstore_idx:reduce_size + dstore_idx] = features.detach().cpu().numpy().astype(np.float32)
                    dstore_vals[dstore_idx:reduce_size + dstore_idx] = target.cpu().numpy().astype(np.int)
                

                if args.chunk_size>1 and args.save_decoder_states:
                    dstore_decoder_states[decoder_states_size : decoder_states.size(0)+decoder_states_size] = decoder_states.detach().cpu().numpy().astype(np.float16)

                    
                    if i==0:
                        decoder_states_map = [[n+j for j in range(args.chunk_size)] for n in range(features.size(0))]
                    else:
                        decoder_states_map.extend([[decoder_states_size+n+j for j in range(args.chunk_size)]  for n in range(features.size(0))])

                    decoder_states_size += decoder_states.size(0)

                dstore_idx += reduce_size

            if args.save_decoder_states and args.chunk_size>1:
                
                with open(args.dstore_mmap+'/decoder_states_map.pkl','wb') as f:
                    pickle.dump(decoder_states_map,f)

                if args.pca_decoder_states>0:
                    pca = PCA(n_components=args.pca_decoder_states)
                    print('pca')
                    random_sample = np.random.choice(np.arange(decoder_states_size), size=[min(10000, decoder_states_size)],replace=False)
                    print('random sample')
                    pca.fit(dstore_decoder_states[random_sample])
                    pickle.dump(pca, open(args.dstore_mmap + '/pca_'+ str(args.pca_decoder_states),'wb'))
                    print('fit')
                    for b in range(0,decoder_states_size, 100000):
                        decoder_states = dstore_decoder_states[b:b+100000]
                        if b==0:
                            pca_decoder_states = torch.FloatTensor(pca.transform(decoder_states)).half()
                        else:
                            pca_decoder_states = torch.cat([pca_decoder_states, torch.FloatTensor(pca.transform(decoder_states)).half()],0)
                        #print(pca_decoder_states.shape)
                    print('save')
                    torch.save(pca_decoder_states, args.dstore_mmap + '/decoder_states_pca_'+str(args.pca_decoder_states))
                else:
                    for b in range(0,decoder_states_size,100000):
                        if b==0:
                            save_decoder_states = torch.FloatTensor(dstore_decoder_states[b:b+100000]).half()
                        else:
                            save_decoder_states = torch.cat([save_decoder_states, torch.FloatTensor(dstore_decoder_states[b:b+100000]).half()],0)

                    torch.save(save_decoder_states, args.dstore_mmap + '/decoder_states')

    print('Creating datastore took: {} s'.format(time.time() - start))

def cli_main():
    parser = options.get_save_datastore_parser()
    args = options.parse_args_and_arch(parser)

    # only override args that are explicitly given on the command line
    override_parser = options.get_save_datastore_parser()
    override_args = options.parse_args_and_arch(override_parser, suppress_defaults=True)

    #distributed_utils.call_main(args, main, override_args=override_args)
    main(args, override_args=override_args)

if __name__ == "__main__":
    cli_main()

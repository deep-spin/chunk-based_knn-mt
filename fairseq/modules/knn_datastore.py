import torch
import faiss
import numpy as np
from torch_scatter import scatter
import time
import math
import faiss.contrib.torch_utils
import pickle
from sklearn.decomposition import PCA


from threading import Thread


class KNN_Dstore(object):

    def __init__(self, args, trg_vocab_size):

        self.half = args.decoder.fp16
        self.dimension = args.decoder.embed_dim
        self.dstore_size = args.dstore_size
        self.metric_type = args.faiss_metric_type
        self.sim_func = args.knn_sim_func
        self.dstore_fp16 = args.dstore_fp16
        self.use_gpu_to_search = args.use_gpu_to_search
        self.vocab_size = trg_vocab_size

        self.chunk_size = args.chunk_size
        self.simplistic = args.simplistic
        self.unsupervised = args.unsupervised
        self.all_neighbours = args.all_neighbours
        self.keep_previous_neighbours = args.keep_previous_neighbours
        self.previous_neighbours_threshold = args.previous_neighbours_threshold
        self.pca_decoder_states = args.pca_decoder_states

        self.supervised = args.supervised

        self.retrieval_threshold = args.retrieval_threshold

        self.use_local_faiss = args.use_local_faiss
        self.res = faiss.StandardGpuResources() 

        self.on_the_fly = args.on_the_fly
        self.on_the_fly_n_updates = args.on_the_fly_n_updates
        self.on_the_fly_keys_size = args.on_the_fly_keys_size
        self.on_the_fly_reindexing = args.on_the_fly_reindexing
        self.start_on_the_fly_keys_vals = 0
        
        if args.knn_temperature_value_2 != 0:
            self.knn_temperature_value_2 = args.knn_temperature_value_2
        else:
            self.knn_temperature_value_2 = args.knn_temperature_value

        self.dstore_filename = args.dstore_filename
        self.index_pca = args.index_pca

        self.index = self.setup_faiss(args)

        self.time_for_retrieve = 0.
        self.retrieve_count = 0.
        self.time_for_setup_prob = 0.

        # set lambda
        self.set_lambda(args)

        # set temperature
        self.temperature_type = args.knn_temperature_type
        if self.temperature_type == 'fix':
            self.temperature = args.knn_temperature_value
        elif self.temperature_type == 'trainable':
            self.temperature = None
        else:
            self.temperature = None

        self.k = args.k

        self.mask_for_label_count = self.generate_label_count_mask(args.k)

        self.knn_tgt_prob=None

        self.knn_vary_c = args.knn_vary_c

    def generate_neighbor_mask(self, max_k):

        # [1, 1000, 1000]
        # [1, 1,    1000]
        # [1, 1,    1   ]
        k_mask = torch.empty((max_k, max_k)).fill_(999.)
        k_mask = torch.triu(k_mask, diagonal=1) + 1

        # we only select 2's power here
        # [1 - 1, 2 - 1, 4 - 1, 8 - 1, ...]
        power_index = torch.tensor([pow(2, i) - 1 for i in range(0, int(math.log(self.max_k, 2)) + 1)])
        k_mask = k_mask[power_index]

        k_mask.requires_grad = False
        if torch.cuda.is_available():
            k_mask = k_mask.cuda()

        return k_mask

    def generate_label_count_mask(self, max_k):

        # [0, 1, 1]
        # [0, 0, 1]
        # [0, 0, 0]
        mask_for_label_count = torch.empty((max_k, max_k)).fill_(1)
        mask_for_label_count = torch.triu(mask_for_label_count, diagonal=1).bool()

        if torch.cuda.is_available():
            mask_for_label_count = mask_for_label_count.cuda()

        mask_for_label_count.requires_grad = False

        return mask_for_label_count

    def get_label_count_segment(self, tgt_idx: torch.Tensor, relative=False):  # [B, S, K]
        """
        This function return the label counts for different range of k nearest neighbor
        [[0:0], [0:1], [0:2], ..., [0:K-1]]

        """

        B, S, K = tgt_idx.size()

        expand_tgt_idx = tgt_idx.unsqueeze(-2).expand(B, S, K, K)
        expand_tgt_idx = expand_tgt_idx.masked_fill(self.mask_for_label_count, value=-1)

        labels_sorted, _ = expand_tgt_idx.sort(dim=-1)  # [B, S, K, K]
        labels_sorted[:, :, :, 1:] *= ((labels_sorted[:, :, :, 1:] - labels_sorted[:, :, :, :-1]) != 0).long()
        retrieve_label_counts = labels_sorted.ne(0).sum(-1)  # [B, S, K]
        retrieve_label_counts[:, :, :-1] -= 1

        # if we want relative label count, i.e [1, 2, 3, 3, 4] -> [1, 1, 1, 0, 1]
        if relative:
            retrieve_label_counts[:, :, 1:] = retrieve_label_counts[:, :, 1:] - retrieve_label_counts[:, :, :-1]

        return retrieve_label_counts

    def get_label_count(self, tgt_idx: torch.Tensor):
        """
        This only return total label count for all neighbors
        """
        tgt_sorted, _ = tgt_idx.sort(dim=-1)
        tgt_sorted[:, :, 1:] *= ((tgt_sorted[:, :, 1:] - tgt_sorted[:, :, :-1]) != 0).long()
        retrieve_label_counts = tgt_sorted.ne(0).sum(-1).unsqueeze(-1)  # [B, S, 1]

        return retrieve_label_counts

    def set_lambda(self, args):

        if not hasattr(args, 'knn_lambda_type'):
            return

        self.lambda_type = args.knn_lambda_type

        if self.lambda_type == 'fix':
            self.lambda_value = args.knn_lambda_value

        if self.lambda_type == 'trainable':
            self.lambda_value = None  # not generate lambda value in this class

    def get_faiss_centroids(self):
        return self.index.quantizer.reconstruct_n(0, self.index.nlist)

    def get_lambda(self, step=None, distance=None):

        if self.lambda_type == 'fix':

            return self.lambda_value

        elif self.lambda_type == 'trainable':

            return None

    def get_temperature(self):

        if self.temperature_type == 'fix':
            return self.temperature
        else:
            return None

    def setup_faiss(self, args):

        if not args.dstore_filename:
            raise ValueError('Cannot build a datastore without the data.')

       
        start = time.time()
        index = faiss.read_index(args.dstore_filename + args.index_pca + 'knn_index', faiss.IO_FLAG_ONDISK_SAME_DIR)

        if self.use_gpu_to_search:
            print('put index from cpu to gpu')
            res = faiss.StandardGpuResources()
            self.res = res
            co = faiss.GpuClonerOptions()
            co.useFloat16 = True
            index = faiss.index_cpu_to_gpu(res, 0, index, co)

        print('Reading datastore took {} s'.format(time.time() - start))
        print('the datastore is {}, size is {}, and dim is {} '.format(args.dstore_filename, self.dstore_size, self.dimension))

        index.nprobe = args.probe

        if args.dstore_fp16:
            print('Keys are fp16 and vals are int32')
            if not args.no_load_keys:
                self.keys = np.memmap(args.dstore_filename + 'keys.npy', dtype=np.float16, mode='r',shape=(self.dstore_size, self.dimension))
            self.vals = np.memmap(args.dstore_filename + 'vals.npy', dtype=np.int, mode='r',shape=(self.dstore_size, self.chunk_size))
        else:
            print('Keys are fp32 and vals are int32')
            if not args.no_load_keys:
                self.keys = np.memmap(args.dstore_filename + 'keys.npy', dtype=np.float32, mode='r',shape=(self.dstore_size, self.dimension))

            self.vals = np.memmap(args.dstore_filename + 'vals.npy', dtype=np.int, mode='r',shape=(self.dstore_size, self.chunk_size))
            
            print('loaded keys and vals')
            if self.chunk_size > 1 and self.unsupervised:
                if self.pca_decoder_states>0:
                    self.decoder_states = torch.load(args.dstore_filename + 'decoder_states_pca_' + str(self.pca_decoder_states))
                    self.decoder_pca = pickle.load(open(args.dstore_filename + 'pca_' + str(self.pca_decoder_states),'rb'))
                else:
                    self.decoder_states = torch.load(args.dstore_filename + 'decoder_states')
                
                print('loaded decoder states')

                with open(args.dstore_filename + 'decoder_states_map.pkl', 'rb') as f:
                    self.decoder_states_map = pickle.load(f)
                self.decoder_states_map = torch.LongTensor(self.decoder_states_map)

        if self.on_the_fly_n_updates>0:
            self.first_datastore_update = True

            self.on_the_fly_keys = np.memmap(args.dstore_filename + 'on_the_fly_keys.npy',
                                            dtype=np.float16 if args.dstore_fp16 else np.float32, mode='r',
                                            shape=(self.on_the_fly_keys_size, self.dimension))
            
            self.on_the_fly_vals_from_memmap = np.memmap(args.dstore_filename + 'on_the_fly_vals.npy', dtype=np.int, mode='r', 
                                            shape=(self.on_the_fly_keys_size, self.chunk_size))
            self.on_the_fly_vals = np.zeros((self.on_the_fly_keys_size, self.chunk_size), dtype=np.int)
            self.on_the_fly_vals = self.on_the_fly_vals_from_memmap[:]
            self.on_the_fly_vals = self.on_the_fly_vals.astype(np.int)

            with open(args.dstore_filename + 'on_the_fly_key_vals_map.pkl', 'rb') as f:
                self.on_the_fly_key_vals_map = pickle.load(f)

            if self.chunk_size > 1:
                self.on_the_fly_decoder_states = torch.load(args.dstore_filename + 'on_the_fly_decoder_states_pca_64')
                with open(args.dstore_filename + 'on_the_fly_decoder_states_map.pkl', 'rb') as f:
                    self.on_the_fly_decoder_states_map = pickle.load(f)
                self.on_the_fly_decoder_states_map = torch.LongTensor(self.on_the_fly_decoder_states_map)


        # If you wish to load all the keys into memory
        # CAUTION: Only do this if your RAM can handle it!
        if args.move_dstore_to_mem:
            print('Loading to memory...')
            start = time.time()

            if not args.no_load_keys:
                del self.keys
                self.keys_from_memmap = np.memmap(args.dstore_filename + '/keys.npy',
                                                  dtype=np.float16 if args.dstore_fp16 else np.float32, mode='r',
                                                  shape=(self.dstore_size, self.dimension))
                self.keys = np.zeros((self.dstore_size, self.dimension),
                                     dtype=np.float16 if args.dstore_fp16 else np.float32)
                self.keys = self.keys_from_memmap[:]
                self.keys = self.keys.astype(np.float16 if args.dstore_fp16 else np.float32)

            del self.vals
            self.vals_from_memmap = np.memmap(args.dstore_filename + 'vals.npy', dtype=np.int, mode='r', shape=(self.dstore_size, self.chunk_size))
            self.vals = np.zeros((self.dstore_size, self.chunk_size), dtype=np.int)
            self.vals = np.array(self.vals_from_memmap[:])
            self.vals = self.vals.astype(np.int)
                


            if self.use_gpu_to_search:
                self.vals = torch.from_numpy(self.vals)
                if torch.cuda.is_available():
                    print('put vals to gpu')
                    self.vals = self.vals.cuda()

            print('Loading to memory took {} s'.format(time.time() - start))

        return index

    def dist_func(self, d, k, q, function=None):

        if not function:
            # Default behavior for L2 metric is to recompute distances.
            # Default behavior for IP metric is to return faiss distances.
            qsize = q.shape
            if self.metric_type == 'l2':
                if torch.cuda.is_available():
                    knns_vecs = torch.from_numpy(self.keys[k]).cuda().view(qsize[0], self.k, -1)
                else:
                    knns_vecs = torch.from_numpy(self.keys[k]).view(qsize[0], self.k, -1)
                if self.half:
                    knns_vecs = knns_vecs.half()
                query_vecs = q.view(qsize[0], 1, qsize[1]).repeat(1, self.k, 1)
                l2 = torch.sum((query_vecs - knns_vecs.detach()) ** 2, dim=2)
                return -1 * l2
            return d

        if function == 'dot':
            qsize = q.shape
            if torch.cuda.is_available():
                return (torch.from_numpy(self.keys[k]).cuda() * q.view(qsize[0], 1, qsize[1])).sum(dim=-1)
            else:
                return (torch.from_numpy(self.keys[k]) * q.view(qsize[0], 1, qsize[1])).sum(dim=-1)

        if function == 'do_not_recomp_l2':
            return -1 * d

        raise ValueError("Invalid knn similarity function!")

    
    def get_knns(self, queries, dstore_idx=None):
        dists, knns = self.index.search(queries, self.k)
        return dists, knns

    def calculate_knn_prob(self, knn_index: torch.Tensor,  # [B, S, K]
                           tgt_index: torch.Tensor,  # [B, S, K]
                           distance: torch.Tensor,  # [B, S, K]
                           queries: torch.Tensor,  # [B, S, H]
                           temperature: torch.Tensor,  # [B, S, 1]
                           ):

        bsz = queries.size(0)
        seq_len = queries.size(1)

        # update the dist and compute each neighbor weight, neg distance
        re_compute_dists = self.dist_func(distance, knn_index, queries, function=self.sim_func)  # [B, S, K]

        scaled_dists = re_compute_dists / temperature
        knn_weight = torch.softmax(scaled_dists, dim=-1).unsqueeze(-1)  # [B, S, K, 1]

        # set the target index for each neighbor
        
        if self.chunk_size>1:
            if self.simplistic:
                self.knn_weight = knn_weight
                self.tgt_index = tgt_index

            knn_tgt_prob = torch.zeros(bsz, seq_len, self.k, self.vocab_size).to(queries.device)  # [B, S, K, Vocab Size]
            tgt_index[0] = tgt_index[0].unsqueeze_(-1)  # [B, S, K, 1]

            scatter(src=knn_weight.float(), out=knn_tgt_prob, index=tgt_index[0], dim=-1)
            prob = knn_tgt_prob.sum(dim=-2)  # [Batch Size, seq len, vocab size]

        else:
            knn_tgt_prob = torch.zeros(bsz, seq_len, self.k, self.vocab_size).to(queries.device)  # [B, S, K, Vocab Size]
            tgt_index = tgt_index.unsqueeze_(1)  # [B, S, K, 1]
            scatter(src=knn_weight.float(), out=knn_tgt_prob, index=tgt_index, dim=-1)
            prob = knn_tgt_prob.sum(dim=-2)  # [Batch Size, seq len, vocab size]

        
        return {'prob': prob}


    def reindex_datastore(self, example_idx):
        time_start = time.time()
        quantizer = faiss.IndexFlatL2(1024)
        self.index = faiss.IndexIVFPQ(quantizer, 1024, 4096, 64, 8)
        self.index.nprobe = 64

        random_sample = np.random.choice(np.arange(self.vals.shape[0]), size=[min(1000000, self.vals.shape[0])],replace=False)
        self.index.train(self.keys[random_sample].astype(np.float32))

        start = 0
        print('adding entries to datastore')
        while start<len(self.vals):
            end = min(start + 1000000, self.vals.shape[0])
            self.index.add_with_ids(self.keys[start:end].astype(np.float32), np.arange(start, end))
            start += 1000000
        
        print('re-indexing took ', time.time()-time_start)


    def update_datastore(self, example_idx):
        print(' updating datastore')
        time_start = time.time()
        #print(example_idx)
        if example_idx%self.on_the_fly_n_updates!=0:
            example_idx = example_idx - example_idx % self.on_the_fly_n_updates
        #print(example_idx)
        end = self.start_on_the_fly_keys_vals + np.array(self.on_the_fly_key_vals_map[example_idx-self.on_the_fly_n_updates:example_idx]).sum()
        #print(self.on_the_fly_key_vals_map[example_idx-self.on_the_fly_n_updates:example_idx])
        #print(len(self.on_the_fly_key_vals_map[example_idx-self.on_the_fly_n_updates:example_idx]))
        #print(np.array(self.on_the_fly_key_vals_map[example_idx-self.on_the_fly_n_updates:example_idx]).sum())
        #print(torch.FloatTensor(self.on_the_fly_keys[self.start_on_the_fly_keys_vals : end]).shape)
        #print(self.start_on_the_fly_keys_vals, end)
        self.index.add(torch.FloatTensor(self.on_the_fly_keys[self.start_on_the_fly_keys_vals : end]))
        if self.on_the_fly_reindexing:
            self.keys = np.concatenate((self.keys, self.on_the_fly_keys[self.start_on_the_fly_keys_vals : end]), 0)    
        self.vals = np.concatenate((self.vals, self.on_the_fly_vals[self.start_on_the_fly_keys_vals : end]), 0)

        if self.chunk_size>1:
            if self.first_datastore_update:
                last = self.decoder_states_map[-1][-1].item()
                self.last_first = self.decoder_states_map[-1][0].item()
                self.decoder_states = self.decoder_states[:last+1]

            add_decoder_states_map = self.on_the_fly_decoder_states_map[self.start_on_the_fly_keys_vals : end]

            self.decoder_states_map = torch.cat([self.decoder_states_map, add_decoder_states_map + self.last_first+1], 0)   

            add_decoder_states = self.on_the_fly_decoder_states[self.decoder_states.size(0)-self.last_first-1: add_decoder_states_map[-1][-1]+1]

            self.decoder_states = torch.cat([self.decoder_states, add_decoder_states],0)

        self.start_on_the_fly_keys_vals = end 
        self.first_datastore_update = False
        print('updating datastore took ', time.time()-time_start)


    def retrieve(self, queries, dstore_idx=None, c=None):

        # queries  are [Batch, seq len, Hid Size]

        # retrieve
        bsz = queries.size(0)
        seq_len = queries.size(1)

        dists, knns = self.get_knns(queries.contiguous().view(-1, queries.size(-1)).cpu())  # [Batch * seq len, K]

        if self.knn_vary_c:
            dists = dists[:,:c]
            knns = knns[:,:c]

        if not self.use_gpu_to_search:
            tgt_idx = torch.from_numpy(self.vals[knns]).to(queries.device).squeeze(-1)  # [Batch size * Seq len, K]
        else:
            tgt_idx = self.vals[knns].to(queries.device).squeeze(-1)

        if self.chunk_size > 1 and not self.supervised:
            tgt_idx_=tgt_idx
            tgt_idx={}
            if self.simplistic:
                for i in range(self.chunk_size):
                    tgt_idx[i] = tgt_idx_[:,:,i].view(bsz, seq_len, -1)  # [B, S, K]
            elif self.unsupervised:
                tgt_idx[0] = tgt_idx_[:,:,0].view(bsz, seq_len, -1)  # [B, S, K]
                if self.all_neighbours:
                    tgt_idx[1] = tgt_idx_[:,:,:].contiguous().view(bsz, seq_len, -1)
                else:
                    tgt_idx[1] = tgt_idx_[:,:,1:].contiguous().view(bsz, seq_len, -1)

        if torch.cuda.is_available():
            dists = dists.view(bsz, seq_len, -1).cuda()  # [Batch, Seq len, k]
            knns = knns.view(bsz, seq_len, -1).cuda()
        else:
            dists = dists.view(bsz, seq_len, -1)  # [Batch, Seq len, k]
            knns = knns.view(bsz, seq_len, -1)

        return {'distance': dists, 'knn_index': knns, 'tgt_index': tgt_idx}


    def compute_prob_simplistic(self, beam_ids, idx):
        
        bsz = int(beam_ids.size(0)*beam_ids.size(1))

        if idx==1:
            self.knn_tgt_prob = torch.zeros(bsz, 1, self.k, self.vocab_size).cuda()  # [B, S, K, Vocab Size]
        else:
            self.knn_tgt_prob[:,:,:,:]=0
            self.knn_tgt_prob=self.knn_tgt_prob[:bsz]

        self.knn_weight=self.knn_weight[beam_ids.view(-1)]
        tgt_index=self.tgt_index[idx][beam_ids].view(bsz,1,self.k).unsqueeze(-1)

        scatter(src=self.knn_weight.float(), out=self.knn_tgt_prob, index=tgt_index, dim=-1)
        prob = self.knn_tgt_prob.sum(dim=-2)  # [Batch Size, seq len, vocab size]

        return prob


    def compute_prob_unsupervised(self, queries, knn_index, neighbors, beam_ids, chunk_start=False, sent_start=False, tgt_dict=None):

        batch_size=8
        beam_size=5
        bsz = queries.size(0)
        seq_len = queries.size(1)

        if self.pca_decoder_states>0:
            if queries.size(0)==1 and queries.size(1)==1:
                queries = torch.FloatTensor(self.decoder_pca.transform(queries.squeeze(0).cpu())).cuda().unsqueeze(1)
            else:
                queries = torch.FloatTensor(self.decoder_pca.transform(queries.squeeze().cpu())).cuda().unsqueeze(1)
                
        if chunk_start:
            """
            print('\n')
            print('new neighbors: ', neighbors.shape)
            new_tokens_=[tgt_dict[idx] for j in neighbors for idx in j ]
            new_tokens=[]
            while len(new_tokens_)>1:
                new_tokens.append(new_tokens_[0:16])
                new_tokens_=new_tokens_[16:]
            i=0
            for s in new_tokens:
                i+=1
                print(list(filter(lambda a: a != '<pad>', s)))
                if i%8==0:
                    print('\n')
            """
            if self.all_neighbours:
                if self.keep_previous_neighbours and not sent_start:

                    neighbors = torch.cat([self.neighbors.view(-1), neighbors.view(-1)], -1)

                    if self.use_local_faiss:
                        self.local_index.add(self.decoder_states[self.decoder_states_map[knn_index][:,:,:]].cuda().view(-1, queries.size(-1)).float())
                    else:
                        decoder_states = self.decoder_states[self.decoder_states_map[knn_index][:,:,:]].cuda().view(1, -1, queries.size(-1)).float()
                        self.decoder_states_ = torch.cat([self.decoder_states_, decoder_states] , 1)

                    
                    if self.previous_neighbours_threshold>0:
                        threshold = int(self.previous_neighbours_threshold*self.k*self.chunk_size*batch_size*beam_size)
                        if self.use_local_faiss and self.local_index.ntotal>threshold:
                            local_index = faiss.index_gpu_to_cpu(self.local_index)
                            local_index.remove_ids(np.arange(0, local_index.ntotal - threshold))
                            self.local_index = faiss.index_cpu_to_gpu(self.res, 0, local_index)
                        elif not self.use_local_faiss:
                            self.decoder_states_ = self.decoder_states_[:,-threshold:]
                        
                        neighbors = neighbors[-threshold:]

                    self.neighbors = neighbors

                    #print('neighbors cache: ', self.neighbors.shape)
                    #print(self.neighbors)
                    #print('\n\n')

                else:
                    if self.all_neighbours:
                        if self.use_local_faiss:
                            local_index = faiss.IndexFlatL2(self.pca_decoder_states)
                            self.local_index = faiss.index_cpu_to_gpu(self.res, 0, local_index)
                            self.local_index.add(self.decoder_states[self.decoder_states_map[knn_index][:,:,:]].cuda().view(-1, queries.size(-1)).float())
                        else:
                            self.decoder_states_ = self.decoder_states[self.decoder_states_map[knn_index][:,:,:]].cuda().view(1, -1, queries.size(-1)).float()
                        self.neighbors = neighbors

                    else:
                        self.decoder_states_ = self.decoder_states[self.decoder_states_map[knn_index][:,:,1:]].cuda().view(1, -1, queries.size(-1)).float()
            else:       
                self.decoder_states_ = self.decoder_states[self.decoder_states_map[knn_index][:,:,1:]].cuda().view(bsz, -1, queries.size(-1)).float()
            self.knn_tgt_prob = torch.zeros(bsz, seq_len, self.k, self.vocab_size).to(queries.device)  # [B, S, K, Vocab Size]
        else:
            if not self.all_neighbours:
                self.decoder_states_ = self.decoder_states_[beam_ids.view(-1)]
            elif self.keep_previous_neighbours:
                neighbors = self.neighbors

            self.knn_tgt_prob[:,:,:,:]=0
            self.knn_tgt_prob=self.knn_tgt_prob[:bsz]

        if not self.use_local_faiss:
            dists = torch.cdist(queries, self.decoder_states_, p=2)
        else:
            dists, indeces = self.local_index.search(queries.squeeze(1), self.k)
            dists = torch.sqrt(dists) 

        if self.retrieval_threshold>0 and self.retrieval_threshold<dists.min(-1).values.mean():
            return None
        
        if self.all_neighbours:
            neighbors = neighbors.view(1,-1).repeat(bsz,1)

        if (self.chunk_size>2 or self.all_neighbours) and not self.use_local_faiss:
            dists = torch.topk(dists, self.k, largest=False, dim=-1)
            idx = torch.gather(neighbors,1,dists.indices.squeeze(1))
            dists = dists.values
        elif not self.use_local_faiss:
            idx=neighbors
        else:
            idx = torch.gather(neighbors,1, indeces)


        scaled_dists = -1 * dists / self.knn_temperature_value_2
        knn_weight = torch.softmax(scaled_dists, dim=-1)

        if not self.use_local_faiss:
            scatter(src=knn_weight.unsqueeze(-1).float(), out=self.knn_tgt_prob, index=idx.unsqueeze(1).unsqueeze(-1), dim=-1)        
        else:
            scatter(src=knn_weight.unsqueeze(1).unsqueeze(-1).float(), out=self.knn_tgt_prob, index=idx.unsqueeze(1).unsqueeze(-1), dim=-1)        

        prob = self.knn_tgt_prob.sum(dim=-2)  # [Batch Size, seq len, vocab size]

        return prob


    def update_get_knn_seq_prob(self, queries):

        knn_search_result = self.retrieve(queries)

        if self.temperature_type == 'fix':
            final_result = self.calculate_knn_prob(knn_index=knn_search_result['knn_index'],
                                                   tgt_index=knn_search_result['tgt_index'],
                                                   distance=knn_search_result['distance'],
                                                   queries=queries,
                                                   temperature=self.temperature)

            return {'distance': knn_search_result['distance'],
                    'knn_index': knn_search_result['knn_index'],
                    'prob': final_result['prob'],}


if __name__ == "__main__":
    class ARGS:
        fp16 = False
        decoder_embed_dim = 1024
        k = 64
        dstore_size = 524400
        faiss_metric_type = 'do_not_recomp_l2'
        knn_sim_func = 'do_not_recomp_l2'
        dstore_fp16 = True
        knn_temperature = 1.0
        indexfile = ''
        dstore_filename = ''
        no_load_keys = False
        probe = 32
        move_dstore_to_mem = True
        use_gpu_to_search = True
        trg_vocab_size = 42024


    args = ARGS()
    knn_store = KNN_Dstore(args=args, trg_vocab_size=args.trg_vocab_size)

    query = torch.randn(32 * 4, 1024)
    print('query size is {}', query.size())
    # dist, knn_idx = knn_store.get_knns(query)
    # print(dist.shape)  # [10000, 64]
    # print(knn_idx.shape)  # [10000, 64]

    prob = knn_store.get_knn_prob(query)
    # print(prob.max(dim=-1)[0])
    # print(prob.max(dim=-1)[1])

    print('average time for retrieve neighbors, {} s'.format(knn_store.time_for_retrieve / knn_store.retrieve_count))
    print('average time for set the target prob for each neighbor'
          ' (need do scatter operation for (batch size * beam size * k, vocab size) tensor), {} s'
          .format(knn_store.time_for_setup_prob / knn_store.retrieve_count))

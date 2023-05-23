
1) Class Client fun run_epoch arg cur_epoch "a cosa serve" 12/05/2023
2)


cose importanti: 
- il numero di clients va deciso in un altro modo (vedere slack prima domanda per capire) questo per velocizzare il training
- server è allenato solo su client o c'è anche un centralizzato? il dataset del centralizzato nel caso è Emnist


Errore colab in import niid dataset:
errore tentando niid
''Traceback (most recent call last):
  File "/content/MLDL_Project/main.py", line 188, in <module>
    main()
  File "/content/MLDL_Project/main.py", line 182, in main
    train_clients, test_clients = gen_clients(args, train_datasets, test_datasets, model)
  File "/content/MLDL_Project/main.py", line 163, in gen_clients
    clients[i].append(Client(args, ds, model, test_client=i == 1))
  File "/content/MLDL_Project/client.py", line 16, in _init_
    self.model = copy.deepcopy(model)
  File "/usr/lib/python3.10/copy.py", line 172, in deepcopy
    y = _reconstruct(x, memo, *rv)
  File "/usr/lib/python3.10/copy.py", line 271, in _reconstruct
    state = deepcopy(state, memo)
  File "/usr/lib/python3.10/copy.py", line 146, in deepcopy
    y = copier(x, memo)
  File "/usr/lib/python3.10/copy.py", line 231, in _deepcopy_dict
    y[deepcopy(key, memo)] = deepcopy(value, memo)
  File "/usr/lib/python3.10/copy.py", line 172, in deepcopy
    y = _reconstruct(x, memo, *rv)
  File "/usr/lib/python3.10/copy.py", line 297, in _reconstruct
    value = deepcopy(value, memo)
  File "/usr/lib/python3.10/copy.py", line 172, in deepcopy
    y = _reconstruct(x, memo, *rv)
  File "/usr/lib/python3.10/copy.py", line 271, in _reconstruct
    state = deepcopy(state, memo)
  File "/usr/lib/python3.10/copy.py", line 146, in deepcopy
    y = copier(x, memo)
  File "/usr/lib/python3.10/copy.py", line 231, in _deepcopy_dict
    y[deepcopy(key, memo)] = deepcopy(value, memo)
  File "/usr/lib/python3.10/copy.py", line 172, in deepcopy
    y = _reconstruct(x, memo, *rv)
  File "/usr/lib/python3.10/copy.py", line 297, in _reconstruct
    value = deepcopy(value, memo)
  File "/usr/lib/python3.10/copy.py", line 153, in deepcopy
    y = copier(memo)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/parameter.py", line 55, in _deepcopy_
    result = type(self)(self.data.clone(memory_format=torch.preserve_format), self.requires_grad)
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 26.00 MiB (GPU 0; 14.75 GiB total capacity; 13.80 GiB already allocated; 2.81 MiB free; 14.64 GiB reserved in total by PyTorch) If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF''
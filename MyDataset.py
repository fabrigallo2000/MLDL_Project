class MyDataset(Dataset):
    def __init__(self, data, transform_fn):
        self.data = data
        self.transform_fn = transform_fn

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]  # Assumendo che il tuo dataset sia una lista di elementi

        transformed_item = self.transform_fn(item)  # Applica la tua funzione di trasformazione personalizzata

        return transformed_item
